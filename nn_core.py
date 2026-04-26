"""
nn_core.py
----------
Policy-gradient neural network for multi-asset goal-reaching / aspiration
utility maximisation.

Changes vs v1
─────────────
1. tau (time-to-horizon) added as network input — the optimal policy is
   explicitly time-dependent; a stationary net cannot learn bang-bang near T.
2. Browne/Merton supervised pre-training initialises the network in the correct
   basin before policy-gradient starts.
3. Antithetic variates (Z / −Z pairs) halve path-level gradient variance at
   zero extra cost.
4. Curriculum sampling oversamples near-terminal, near-goal paths so the
   critical (w ≈ goal, τ ≈ 0) region is well-covered.
5. Volatility normalisation removed — was a confound versus the FD benchmark.
6. Adaptive sigmoid temperature scales with σ_eff · √T so the smoothing of
   the discontinuous terminal condition matches the actual diffusion width.
7. Patience-based early stopping on held-out validation utility.
8. Ω⁻¹η tangency direction fed as auxiliary inputs (optional) — gives the
   network the correct multi-asset inductive bias without learning it from data.
9. Cosine LR schedule with warm-down.

GPU changes (v3)
────────────────
10. Automatic Mixed Precision (AMP) via torch.amp — ~2× speedup on CUDA;
    automatically disabled on CPU/MPS where it has no benefit.
11. Best-model checkpointing: saves state_dict at peak validation utility and
    restores it before returning — early-stopping now returns the *best* model,
    not the model from the last iteration.
12. torch.compile() option — wraps the network with the inductor backend for
    faster GPU kernels (requires PyTorch ≥ 2.0; silently skipped otherwise).
13. Reproducible seeding: torch.manual_seed + torch.cuda.manual_seed_all
    called at entry so runs are bit-identical across restarts when seed is set.
14. Larger default test-eval paths (4096) for a tighter E[U] estimate on GPU.

Requires PyTorch. Gracefully raises ImportError if not installed.
"""

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for nn_core. Install with: pip install torch"
        )


# ── Numpy helpers (no torch dependency) ─────────────────────────────────────

def _normcdf_np(z):
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _browne_policy_np(w_norm, tau, omega_inv_eta, theta2, d, u):
    """
    Browne/Merton optimal policy — batched numpy.

    pi*(w, τ) = [1 / (1 + log(w/goal) / (θ²τ))] · Ω⁻¹η

    Parameters
    ----------
    w_norm        : (B,) normalised wealth w / goal
    tau           : (B,) or scalar time-to-horizon
    omega_inv_eta : (n,) pre-computed Ω⁻¹η
    theta2        : scalar θ² = ηᵀ Ω⁻¹ η
    d, u          : per-asset bounds

    Returns
    -------
    pi : (B, n)
    """
    tau   = np.maximum(np.broadcast_to(np.asarray(tau, float), w_norm.shape), 1e-10)
    log_r = np.log(np.maximum(w_norm, 1e-10))
    denom = 1.0 + log_r / (theta2 * tau)
    pi_1d = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)    # (B,)
    pi_nd = pi_1d[:, None] * omega_inv_eta[None, :]                # (B, n)
    return np.clip(pi_nd, d, u)


def _browne_V_np(w_norm, tau, theta2):
    """
    Browne value function (goal normalised to 1).
    V(w, τ) = Φ((log w + (θ² − ½θ²)τ) / (θ√τ))
    """
    theta = math.sqrt(max(theta2, 1e-12))
    tau   = np.maximum(np.asarray(tau, float), 1e-10)
    z     = (np.log(np.maximum(np.asarray(w_norm, float), 1e-10))
             + 0.5 * theta2 * tau) / (theta * np.sqrt(tau))
    return _normcdf_np(z)


# ── Network architecture ─────────────────────────────────────────────────────
# Guard the class definition so the module imports cleanly without PyTorch.
_ModuleBase = nn.Module if HAS_TORCH else object


class PolicyNet(_ModuleBase):
    """
    Policy network: (w_norm, tau [, tangency_direction]) → n-asset weights.

    input_dim = 2     : [w/goal,  τ/T]                   minimum useful input
    input_dim = 2 + n : [w/goal,  τ/T,  Ω⁻¹η / ‖Ω⁻¹η‖]  full tangency hint

    The tanh-affine output layer hard-clips into [d, u] by construction.
    """

    def __init__(self, n_assets=5, hidden=128, d=-5.0, u=3.0, input_dim=2):
        _require_torch()
        super().__init__()
        self.d         = d
        self.u         = u
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,    hidden), nn.Tanh(),
            nn.Linear(hidden,    hidden), nn.Tanh(),
            nn.Linear(hidden,    n_assets),
        )

    def forward(self, features):
        """features : (batch, input_dim)"""
        raw = self.net(features)
        return 0.5 * (self.u - self.d) * torch.tanh(raw) + 0.5 * (self.u + self.d)


# ── Training ─────────────────────────────────────────────────────────────────

class _null_ctx:
    """No-op context manager — used when AMP is disabled."""
    def __enter__(self):  return self
    def __exit__(self, *_): pass


def train_policy_net(
    mu_vec, Omega_mat, r, T=1.0,
    w0=1.0, goal_mult=1.10,
    n_paths=1024, n_iters=500,
    hidden=128, lr=3e-3, n_steps=40,
    d=-5.0, u=3.0,
    utility='goalreach',
    asp_p=0.5, asp_c1=1.2, asp_R=1.0,
    # ── v2 parameters ───────────────────────────────────────────────────
    pretrain_iters=200,       # supervised Browne warm-start iterations (0 = skip)
    p_curriculum=0.30,        # fraction of paths starting near goal boundary
    antithetic=True,          # antithetic variates: run Z and −Z pairs
    use_tangency_input=True,  # feed normalised Ω⁻¹η direction as extra inputs
    patience=80,              # early-stopping patience (iters with no improvement)
    # ── v3 GPU parameters ───────────────────────────────────────────────
    seed=None,                # integer → reproducible torch RNG; None = random
    use_amp=True,             # Automatic Mixed Precision (CUDA only; auto-off on CPU/MPS)
    compile_model=False,      # torch.compile() the network (PyTorch ≥ 2.0, CUDA only)
    test_paths=4096,          # held-out paths for the post-training E[U] estimate
    # ────────────────────────────────────────────────────────────────────
    device=None, verbose=True,
):
    """
    Train a PolicyNet via policy gradient (maximise expected terminal utility).

    Returns
    -------
    net          : trained PolicyNet (eval mode), weights restored to best-val checkpoint
    histories    : dict with keys:
                     "train"     — list[float], E[U] per training iteration
                     "val"       — list[float], E[U] on held-out paths (every 20 iters)
                     "val_iters" — list[int],   iteration indices for val
                     "test_u"    — float,        post-training held-out E[U]
    """
    _require_torch()

    # ── Device selection ─────────────────────────────────────────────────────
    if device is None:
        device = (
            torch.device('mps')  if torch.backends.mps.is_available()  else
            torch.device('cuda') if torch.cuda.is_available()          else
            torch.device('cpu')
        )
    elif isinstance(device, str):
        device = torch.device(device)

    # ── Reproducible seeding ─────────────────────────────────────────────────
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    # ── AMP scaler: only meaningful on CUDA; silently disabled elsewhere ──────
    _use_amp = use_amp and (device.type == 'cuda')
    scaler   = torch.amp.GradScaler('cuda') if _use_amp else None

    n    = len(mu_vec)
    goal = w0 * goal_mult
    eta  = np.asarray(mu_vec, float) - r

    # ── Pre-compute fixed analytical quantities ──────────────────────────────
    omega_inv_eta = np.linalg.solve(Omega_mat, eta)                   # (n,)
    theta2        = max(float(np.dot(eta, omega_inv_eta)), 1e-12)     # max Sharpe²
    tang_norm     = omega_inv_eta / (np.linalg.norm(omega_inv_eta) + 1e-12)

    # Adaptive sigmoid temperature: proportional to diffusion width σ_eff·√T
    eff_sigma = math.sqrt(max(theta2, 1e-12))
    sig_temp  = max(0.30 * eff_sigma * math.sqrt(T), 0.02)

    # ── Build network ────────────────────────────────────────────────────────
    input_dim = 2 + (n if use_tangency_input else 0)
    net = PolicyNet(n_assets=n, hidden=hidden, d=d, u=u,
                    input_dim=input_dim).to(device)

    # Optional torch.compile() — requires PyTorch ≥ 2.0 and CUDA
    if compile_model and device.type == 'cuda':
        try:
            net = torch.compile(net)
            if verbose:
                print("  [compile] torch.compile() applied (inductor backend)")
        except Exception as e:
            if verbose:
                print(f"  [compile] torch.compile() skipped: {e}")

    opt = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(n_iters, 1), eta_min=lr * 0.05
    )

    # ── Fixed tensors ────────────────────────────────────────────────────────
    mu_t   = torch.tensor(mu_vec,  dtype=torch.float32, device=device)
    Sig_t  = torch.tensor(
        np.linalg.cholesky(np.asarray(Omega_mat, float) + 1e-10 * np.eye(n)),
        dtype=torch.float32, device=device,
    )
    r_t    = torch.tensor(float(r), dtype=torch.float32, device=device)
    tang_t = torch.tensor(tang_norm.astype(np.float32), device=device)  # (n,)

    dt   = T / n_steps
    sqdt = math.sqrt(dt)

    # ── Feature builder ──────────────────────────────────────────────────────
    def make_features(W, tau_val):
        """
        W       : (B, 1) wealth tensor
        tau_val : float time-to-horizon (seconds)
        Returns : (B, input_dim) feature tensor
        """
        w_norm = (W / goal).clamp(1e-6, 10.0)
        tau_n  = torch.full(
            (W.shape[0], 1), float(tau_val) / T,
            dtype=torch.float32, device=device,
        )
        parts = [w_norm, tau_n]
        if use_tangency_input:
            parts.append(tang_t.unsqueeze(0).expand(W.shape[0], -1))
        return torch.cat(parts, dim=1)

    # ────────────────────────────────────────────────────────────────────────
    # Phase 1: Supervised Browne/Merton pre-training
    # ────────────────────────────────────────────────────────────────────────
    if pretrain_iters > 0:
        if verbose:
            print("  [pre-train] supervised Browne initialisation ...")
        pre_opt = optim.Adam(net.parameters(), lr=lr * 2.0)
        rng     = np.random.default_rng(42)

        for pit in range(pretrain_iters):
            B      = 512
            w_s    = rng.uniform(0.5, 2.0, size=B).astype(np.float32)   # w/goal
            tau_s  = rng.uniform(dt, T,    size=B).astype(np.float32)

            # Browne target — correct answer for this (w, τ) pair
            pi_tgt = _browne_policy_np(w_s, tau_s, omega_inv_eta, theta2, d, u)

            # Build feature tensor
            w_feat  = torch.tensor(w_s[:, None],       dtype=torch.float32, device=device)
            tau_n_f = torch.tensor((tau_s / T)[:, None], dtype=torch.float32, device=device)
            parts   = [w_feat, tau_n_f]
            if use_tangency_input:
                parts.append(tang_t.unsqueeze(0).expand(B, -1))
            feat = torch.cat(parts, dim=1)

            pi_pred   = net(feat)
            pi_target = torch.tensor(pi_tgt, dtype=torch.float32, device=device)
            pre_loss  = ((pi_pred - pi_target) ** 2).mean()

            pre_opt.zero_grad()
            pre_loss.backward()
            pre_opt.step()

        if verbose:
            print(f"  [pre-train] done — final MSE: {pre_loss.item():.5f}")

    # ────────────────────────────────────────────────────────────────────────
    # Phase 2: Policy-gradient training
    # ────────────────────────────────────────────────────────────────────────
    loss_history       = []   # E[U] on training paths, every iteration
    val_history        = []   # E[U] on held-out paths, every 20 iters
    val_iters          = []   # iteration indices for val_history (for x-axis alignment)
    best_val_utility   = -math.inf
    best_state_dict    = None  # best-val checkpoint (CPU copy to save GPU memory)
    no_improve_count   = 0

    # With antithetic we simulate base_paths forward and mirror with −Z,
    # giving n_paths = 2 * base_paths total at the cost of one forward pass.
    base_paths = n_paths // 2 if antithetic else n_paths

    for it in range(n_iters):

        # ── Curriculum initialisation ────────────────────────────────────────
        # A fraction p_curriculum of paths starts uniformly in [0.70·goal, 1.05·goal]
        # so the near-goal boundary is always well-sampled regardless of w0.
        n_curr = int(base_paths * p_curriculum)
        n_reg  = base_paths - n_curr
        w_reg  = torch.full((n_reg,  1), float(w0), dtype=torch.float32, device=device)
        w_curr = goal * (0.70 + 0.35 * torch.rand(n_curr, 1, device=device))
        W_base = torch.cat([w_reg, w_curr], dim=0)            # (base_paths, 1)

        W = torch.cat([W_base, W_base.clone()], dim=0) if antithetic else W_base

        # ── Simulate forward (wrapped in AMP autocast on CUDA) ───────────────
        ctx = torch.amp.autocast('cuda') if _use_amp else _null_ctx()
        with ctx:
            for step in range(n_steps):
                tau_val = T - step * dt                        # time-to-horizon
                feat    = make_features(W, tau_val)
                pi      = net(feat).clamp(d, u)               # box constraint

                # Antithetic variates: first half uses Z, second half uses −Z
                if antithetic:
                    Z_base = torch.randn(base_paths, n, device=device)
                    Z      = torch.cat([Z_base, -Z_base], dim=0)
                else:
                    Z = torch.randn(n_paths, n, device=device)

                dS     = mu_t * dt + (Z @ Sig_t.T) * sqdt
                bond   = r_t * dt
                excess = (pi * (dS - bond)).sum(1, keepdim=True)
                W      = (W * (1.0 + bond + excess)).clamp(min=1e-6)

            # ── Terminal utility ─────────────────────────────────────────────
            w_norm = W.squeeze() / goal
            if utility == 'goalreach':
                # Adaptive temperature: matches diffusion width near terminal time
                U_term = torch.sigmoid((w_norm - 1.0) / sig_temp)
            elif utility == 'aspiration':
                U_term = torch.where(
                    w_norm < asp_R,
                    w_norm ** asp_p / asp_p,
                    asp_c1 * w_norm ** asp_p / asp_p,
                )
            else:
                raise ValueError(
                    f"utility must be 'goalreach' or 'aspiration', got {utility!r}"
                )

            # ── Loss: maximise E[U] ──────────────────────────────────────────
            loss = -U_term.mean()

        opt.zero_grad()
        if _use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        scheduler.step()

        eu = -loss.item()
        loss_history.append(eu)

        # ── Validation & early stopping (every 20 iters) ─────────────────────
        if it % 20 == 0:
            with torch.no_grad():
                W_val = torch.full((512, 1), float(w0), dtype=torch.float32, device=device)
                for step in range(n_steps):
                    feat_v   = make_features(W_val, T - step * dt)
                    pi_v     = net(feat_v).clamp(d, u)
                    Z_v      = torch.randn(512, n, device=device)
                    dS_v     = mu_t * dt + (Z_v @ Sig_t.T) * sqdt
                    excess_v = (pi_v * (dS_v - r_t * dt)).sum(1, keepdim=True)
                    W_val    = (W_val * (1.0 + r_t * dt + excess_v)).clamp(min=1e-6)
                if utility == 'goalreach':
                    val_u = torch.sigmoid(
                        (W_val.squeeze() / goal - 1.0) / sig_temp
                    ).mean().item()
                else:
                    w_vn = W_val.squeeze() / goal
                    val_u = torch.where(
                        w_vn < asp_R,
                        w_vn ** asp_p / asp_p,
                        asp_c1 * w_vn ** asp_p / asp_p,
                    ).mean().item()

            val_history.append(val_u)
            val_iters.append(it)

            if val_u > best_val_utility + 1e-5:
                best_val_utility  = val_u
                no_improve_count  = 0
                # Save best weights to CPU to avoid holding a second GPU copy
                _raw = net._orig_mod if hasattr(net, '_orig_mod') else net
                best_state_dict   = {
                    k: v.cpu().clone() for k, v in _raw.state_dict().items()
                }
            else:
                no_improve_count += 20

            if verbose and (it + 1) % 100 == 0:
                p_goal = ((W.squeeze() / goal) >= 1.0).float().mean().item()
                print(
                    f"  iter {it+1:4d}  E[U]={eu:.4f}  val_U={val_u:.4f}  "
                    f"P(goal)={p_goal:.3f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        if no_improve_count >= patience:
            if verbose:
                print(
                    f"  early stopping at iter {it+1} "
                    f"(best val_U={best_val_utility:.4f}, "
                    f"no improvement for {patience} iters)"
                )
            break

    # ── Restore best-validation weights before returning ─────────────────────
    if best_state_dict is not None:
        _raw = net._orig_mod if hasattr(net, '_orig_mod') else net
        _raw.load_state_dict(best_state_dict)
        if verbose:
            print(f"  [checkpoint] best-val weights restored  (val_U={best_val_utility:.4f})")

    net.eval()

    # ── Post-training test evaluation ────────────────────────────────────────
    # Run a fresh set of MC paths with a different seed to get an unbiased
    # estimate of E[U] — this is the "test" error, free from any optimism bias
    # that affects the training and validation estimates.
    with torch.no_grad():
        W_test = torch.full((test_paths, 1), float(w0), dtype=torch.float32, device=device)
        for step in range(n_steps):
            feat_t  = make_features(W_test, T - step * dt)
            pi_t    = net(feat_t).clamp(d, u)
            Z_t     = torch.randn(test_paths, n, device=device)
            dS_t    = mu_t * dt + (Z_t @ Sig_t.T) * sqdt
            exc_t   = (pi_t * (dS_t - r_t * dt)).sum(1, keepdim=True)
            W_test  = (W_test * (1.0 + r_t * dt + exc_t)).clamp(min=1e-6)
        if utility == 'goalreach':
            test_u = torch.sigmoid(
                (W_test.squeeze() / goal - 1.0) / sig_temp
            ).mean().item()
        else:
            w_tn = W_test.squeeze() / goal
            test_u = torch.where(
                w_tn < asp_R,
                w_tn ** asp_p / asp_p,
                asp_c1 * w_tn ** asp_p / asp_p,
            ).mean().item()

    histories = {
        "train"     : loss_history,   # list[float], length = n_iters actually run
        "val"       : val_history,    # list[float], length = n_iters // 20
        "val_iters" : val_iters,      # list[int]  — iter index for each val point
        "test_u"    : test_u,         # scalar     — held-out MC E[U] (test_paths paths)
    }
    return net, histories


# ── Policy evaluation ─────────────────────────────────────────────────────────

def nn_policy_weights(net, W_current, goal, tau, T=1.0,
                      omega_inv_eta=None, device=None):
    """
    Query a trained PolicyNet at a single (wealth, time-to-horizon) state.

    Parameters
    ----------
    net           : trained PolicyNet
    W_current     : current wealth (scalar)
    goal          : goal wealth (scalar)
    tau           : time-to-horizon in years
    T             : total horizon — used to normalise tau to [0, 1]
    omega_inv_eta : (n,) pre-computed Ω⁻¹η (required if use_tangency_input=True)

    Returns
    -------
    pi : (n_assets,) numpy array of portfolio weights in [d, u]
    """
    _require_torch()
    if device is None:
        device = next(net.parameters()).device

    w_norm = float(W_current) / max(float(goal), 1e-12)
    tau_n  = float(tau)       / max(float(T),    1e-10)

    parts = [
        torch.tensor([[w_norm]], dtype=torch.float32, device=device),
        torch.tensor([[tau_n]],  dtype=torch.float32, device=device),
    ]
    if omega_inv_eta is not None:
        tang = np.asarray(omega_inv_eta, float)
        tang = tang / (np.linalg.norm(tang) + 1e-12)
        parts.append(
            torch.tensor(tang[None, :], dtype=torch.float32, device=device)
        )

    features = torch.cat(parts, dim=1)
    with torch.no_grad():
        pi = net(features).squeeze().cpu().numpy()
    return pi
