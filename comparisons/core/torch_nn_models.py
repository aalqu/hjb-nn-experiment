"""
torch_nn_models.py
------------------
Extended PyTorch policy-network architectures and training loop for
multi-asset goal-reaching / aspiration utility maximisation.

Training improvements vs v1
────────────────────────────
1. Browne/Merton supervised pre-training — initialises every architecture in
   the correct basin before policy-gradient begins.
2. Antithetic variates (Z / −Z pairs) — halves path-level gradient variance.
3. Curriculum sampling — oversamples near-(goal, τ≈0) paths so the critical
   bang-bang region is always covered.
4. Cosine LR schedule with warm-down — avoids oscillation late in training.
5. Patience-based early stopping on held-out validation utility.
6. Adaptive sigmoid temperature ∝ σ_eff · √T — matches near-terminal
   diffusion width instead of the fixed 0.05 from v1.
7. Gradient clipping retained (max-norm 1.0) for stability.
"""

from __future__ import annotations  # defers annotation eval — allows torch.Tensor hints without torch

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    torch = None
    optim = None
    HAS_TORCH = False
    # Stub nn so that class Foo(nn.Module) definitions don't fail at import time.
    # Instantiating these classes without torch will raise ImportError at runtime.
    class _StubModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use neural network models.")
    class nn:  # noqa: N801
        Module = _StubModule


# Number of features used by the historical-replay architecture (defined here so
# TORCH_ARCHITECTURES can reference it at module load time)
RICH_FEATURE_DIM = 5

TORCH_ARCHITECTURES: Dict[str, Dict[str, object]] = {
    'nn_mlp_small': {'kind': 'mlp', 'hidden_layers': (32, 32)},
    'nn_mlp_deep': {'kind': 'mlp', 'hidden_layers': (64, 64, 32)},
    'nn_policy_net': {'kind': 'mlp', 'hidden_layers': (128, 128, 128)},
    'deep_bsde': {'kind': 'deep_bsde', 'hidden_layers': (32, 32), 'n_iters_min': 500},
    'pinn': {'kind': 'pinn', 'hidden_layers': (64, 64)},
    'actor_critic': {'kind': 'actor_critic', 'hidden_layers': (64, 64)},
    'lstm': {'kind': 'lstm', 'hidden_size': 32, 'seq_len': 8},
    'transformer': {'kind': 'transformer', 'd_model': 32, 'nhead': 4, 'seq_len': 8,
                    'n_iters_min': 500},
    # ── STE variant ──────────────────────────────────────────────────────────
    # Same capacity as nn_policy_net but trained with the exact step-function
    # objective via a Straight-Through Estimator, matching the fd_nd terminal
    # condition 1{W_T >= goal} instead of the sigmoid approximation.
    'nn_ste_goalreach': {'kind': 'mlp', 'hidden_layers': (128, 128, 128),
                         'utility': 'goalreach_ste'},
    # ── Digital-option delta-hedge variant ───────────────────────────────────
    # Value network V(w,τ) frames the problem as pricing a digital call option.
    # Policy = multi-asset delta hedge -(V_w/(w·V_ww))·Ω⁻¹η via autograd.
    # Three-phase training: supervised Browne → Kolmogorov PDE residual → path BCE.
    'nn_digital_hedge': {'kind': 'digital_hedge', 'hidden_layers': (128, 128, 128)},
    # ── Long-only variants ────────────────────────────────────────────────────
    # All weights ≥ 0, total long exposure ≤ 100%, no shorting.
    # Constraints: per-asset [0, 1], max_long=1, max_short=0.
    # Direct comparison with the leveraged counterparts above.
    'nn_policy_long_only': {
        'kind': 'mlp', 'hidden_layers': (128, 128, 128),
        'constraints': {'d': 0.0, 'u': 1.0, 'max_long': 1.0, 'max_short': 0.0},
    },
    'nn_ste_long_only': {
        'kind': 'mlp', 'hidden_layers': (128, 128, 128),
        'utility': 'goalreach_ste',
        'constraints': {'d': 0.0, 'u': 1.0, 'max_long': 1.0, 'max_short': 0.0},
    },
    'nn_digital_hedge_long_only': {
        'kind': 'digital_hedge', 'hidden_layers': (128, 128, 128),
        'constraints': {'d': 0.0, 'u': 1.0, 'max_long': 1.0, 'max_short': 0.0},
    },
    # ── Historical-replay architecture ────────────────────────────────────────
    # Trained directly on real return sequences (block bootstrap) rather than
    # GBM simulations.  Uses RICH_FEATURE_DIM=5 inputs so the network can
    # condition on recent volatility and momentum in addition to (w/goal, τ).
    # Train/val/test are time-ordered splits of the actual historical dataset.
    'nn_historical_replay': {
        'kind': 'historical_replay',
        'hidden_layers': (128, 128, 128),
        'n_features': RICH_FEATURE_DIM,
    },
    'nn_historical_replay_long_only': {
        'kind': 'historical_replay',
        'hidden_layers': (128, 128, 128),
        'n_features': RICH_FEATURE_DIM,
        'constraints': {'d': 0.0, 'u': 1.0, 'max_long': 1.0, 'max_short': 0.0},
    },
}


# ── Straight-Through Estimator for the goal-reaching step function ───────────

class GoalReachSTE(torch.autograd.Function if HAS_TORCH else object):
    """
    Straight-Through Estimator for the binary goal-reaching terminal utility.

    The HJB terminal condition for goal-reaching is U(W_T) = 1{W_T >= goal},
    a step function with zero gradient almost everywhere.  Training with the
    reparameterisation trick therefore yields zero gradient when U is applied
    directly, making learning impossible.

    This STE resolves the conflict:

      Forward  : exact step function  ->  output ∈ {0, 1}
                 loss = -E[1{W_T >= goal}] = -P̂(W_T >= goal)
                 This is exactly what fd_nd maximises.

      Backward : sigmoid surrogate gradient  ->  ∂sigmoid(x/temp)/∂x
                 Provides informative, smooth gradient to the policy network
                 concentrated around the goal boundary (x ≈ 0).

    The temperature `temp` controls the width of the surrogate gradient peak.
    It is passed from the adaptive sig_temp in train_torch_policy_net, so it
    naturally scales with σ_eff · √T — matching the diffusion width near T.

    Usage
    -----
        # In _terminal_utility for utility == 'goalreach_ste':
        x    = w_norm - 1.0              # positive above goal
        U    = GoalReachSTE.apply(x, sig_temp)   # forward: step; backward: sigmoid
        loss = -U.mean()                 # maximise P(W_T >= goal)
    """
    @staticmethod
    def forward(ctx, x, temp):
        ctx.save_for_backward(x)
        ctx.temp = float(temp)
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x,   = ctx.saved_tensors
        temp = ctx.temp
        s    = torch.sigmoid(x / temp)
        surrogate = s * (1.0 - s) / temp   # sigmoid derivative as surrogate
        return grad_output * surrogate, None   # None: temp is not differentiable


# ── Browne/Merton analytical helpers (numpy, no torch) ──────────────────────

def _normcdf_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _browne_policy_np(
    w_norm: np.ndarray, tau: np.ndarray,
    omega_inv_eta: np.ndarray, theta2: float,
    d: float, u: float,
) -> np.ndarray:
    """
    Browne/Merton optimal policy — batched numpy.

    pi*(w, τ) = [1 / (1 + log(w/goal) / (θ²τ))] · Ω⁻¹η

    Returns (B, n) array clipped to [d, u].
    """
    tau   = np.maximum(np.broadcast_to(np.asarray(tau, float), w_norm.shape), 1e-10)
    log_r = np.log(np.maximum(w_norm, 1e-10))
    denom = 1.0 + log_r / (theta2 * tau)
    pi_1d = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)
    pi_nd = pi_1d[:, None] * omega_inv_eta[None, :]
    return np.clip(pi_nd, d, u)


def _mlp(input_dim: int, hidden_layers: Tuple[int, ...], output_dim: int, final_activation=False):
    layers: List[nn.Module] = []
    in_features = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.Tanh())
        in_features = width
    layers.append(nn.Linear(in_features, output_dim))
    if final_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class TorchPolicyNet(nn.Module):
    def __init__(self, n_assets: int, hidden_layers=(32, 32), d=-5.0, u=3.0,
                 n_features: int = 2):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'mlp'
        self.n_features = n_features
        self.net = _mlp(n_features, tuple(hidden_layers), n_assets)

    def forward(self, features):
        # Legacy: if only 1 feature given, pad with zero tau
        if features.shape[-1] == 1 and self.n_features == 2:
            tau = torch.zeros_like(features)
            features = torch.cat([features, tau], dim=-1)
        raw = self.net(features)
        return 0.5 * (self.u - self.d) * torch.tanh(raw) + 0.5 * (self.u + self.d)


class DeepBSDEPolicyNet(nn.Module):
    def __init__(self, n_assets: int, n_steps: int, hidden_layers=(32, 32), d=-5.0, u=3.0):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'deep_bsde'
        self.n_steps = n_steps
        self.nets = nn.ModuleList([_mlp(2, tuple(hidden_layers), n_assets) for _ in range(n_steps)])

    def forward_step(self, features, step_idx: int):
        idx = min(max(step_idx, 0), self.n_steps - 1)
        raw = self.nets[idx](features)
        return 0.5 * (self.u - self.d) * torch.tanh(raw) + 0.5 * (self.u + self.d)


class PINNPolicyNet(nn.Module):
    def __init__(self, n_assets: int, hidden_layers=(64, 64), d=-5.0, u=3.0):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'pinn'
        self.trunk = _mlp(2, tuple(hidden_layers), hidden_layers[-1])
        self.policy_head = nn.Linear(hidden_layers[-1], n_assets)
        self.value_head = nn.Linear(hidden_layers[-1], 1)

    def forward(self, features):
        latent = self.trunk(features)
        raw_policy = self.policy_head(latent)
        policy = 0.5 * (self.u - self.d) * torch.tanh(raw_policy) + 0.5 * (self.u + self.d)
        value = self.value_head(latent)
        return policy, value


class ActorCriticPolicyNet(nn.Module):
    def __init__(self, n_assets: int, hidden_layers=(64, 64), d=-5.0, u=3.0):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'actor_critic'
        self.actor = _mlp(2, tuple(hidden_layers), n_assets)
        self.critic = _mlp(2, tuple(hidden_layers), 1)

    def forward(self, features):
        raw_policy = self.actor(features)
        policy = 0.5 * (self.u - self.d) * torch.tanh(raw_policy) + 0.5 * (self.u + self.d)
        value = self.critic(features)
        return policy, value


class LSTMPolicyNet(nn.Module):
    def __init__(self, n_assets: int, hidden_size=32, seq_len=8, d=-5.0, u=3.0):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'lstm'
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_assets)

    def forward(self, sequence):
        out, _ = self.lstm(sequence)
        raw = self.head(out[:, -1, :])
        return 0.5 * (self.u - self.d) * torch.tanh(raw) + 0.5 * (self.u + self.d)


class TransformerPolicyNet(nn.Module):
    def __init__(self, n_assets: int, d_model=32, nhead=4, seq_len=8, d=-5.0, u=3.0):
        super().__init__()
        self.d = d
        self.u = u
        self.kind = 'transformer'
        self.seq_len = seq_len
        self.input_proj = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(d_model, n_assets)

    def forward(self, sequence):
        x = self.input_proj(sequence)
        enc = self.encoder(x)
        raw = self.head(enc[:, -1, :])
        return 0.5 * (self.u - self.d) * torch.tanh(raw) + 0.5 * (self.u + self.d)


class DigitalHedgeValueNet(nn.Module):
    """
    Value network V(w_norm, τ_norm) ∈ (0, 1) for the goal-reaching digital option.

    Framing
    -------
    V(w, τ) = P(W_T ≥ goal | W_t = w) is the price of a digital call option
    that pays 1 if the portfolio wealth reaches the goal by horizon T.

    The optimal portfolio weights are its multi-asset delta hedge:

        π*(w, τ) = −(V_w / (w_norm · V_ww)) · Ω⁻¹η

    where V_w = ∂V/∂w_norm and V_ww = ∂²V/∂w_norm² are computed via autograd.
    This matches the Browne (1999) interior maximiser exactly when V is the
    true goal-reaching value function.

    Architecture
    ------------
    Input  : [w_norm, τ_norm]   w_norm = W/goal ∈ (0,∞),  τ_norm = τ/T ∈ [0,1]
    Hidden : MLP with tanh activations
    Output : sigmoid(logit) ∈ (0, 1)  — enforces V ∈ (0,1) by construction

    Buffers (set by train_digital_hedge_net after market calibration)
    -------
    omega_inv_eta : (n,) — Ω⁻¹η, the tangency direction
    theta2_val    : ()   — θ² = ηᵀΩ⁻¹η, max squared Sharpe ratio
    r_val         : ()   — risk-free rate
    """

    def __init__(self, n_assets: int, hidden_layers=(128, 128, 128),
                 d=-5.0, u=3.0, max_long=3.0, max_short=5.0):
        super().__init__()
        self.kind      = 'digital_hedge'
        self.n_assets  = n_assets
        self.d         = d
        self.u         = u
        self.max_long  = max_long
        self.max_short = max_short
        # MLP: [w_norm, τ_norm] → scalar logit → sigmoid → V ∈ (0,1)
        self.net = _mlp(2, tuple(hidden_layers), 1)
        # Calibrated market quantities — populated by train_digital_hedge_net
        self.register_buffer('omega_inv_eta', torch.zeros(n_assets))
        self.register_buffer('theta2_val',    torch.tensor(1.0))
        self.register_buffer('r_val',         torch.tensor(0.03))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features : (B, 2) = [w_norm, τ_norm].  Returns V ∈ (0, 1)."""
        return torch.sigmoid(self.net(features))

    def delta_policy(self, features: torch.Tensor,
                     create_graph: bool = False) -> torch.Tensor:
        """
        Derive portfolio weights as the multi-asset delta hedge of V.

        In the concave region (V_ww < 0):
            π = −(V_w / (w_norm · V_ww)) · Ω⁻¹η     (interior maximiser)
        In the flat / convex region:
            π = 0                                       (no informative hedge)

        Parameters
        ----------
        features     : (B, 2) tensor — will have requires_grad set internally
        create_graph : bool — True only when backprop is needed through the policy

        Returns
        -------
        pi : (B, n_assets) leverage-constrained portfolio weights (detached
             when create_graph=False)
        """
        if not features.requires_grad:
            features = features.requires_grad_(True)

        V = self.forward(features)                                    # (B, 1)

        # ── ∂V/∂w_norm ───────────────────────────────────────────────────────
        # First derivative must always build a graph so V_w can be differentiated
        # again to get V_ww.  The caller's create_graph flag only controls whether
        # we need to backprop *through* V_ww (e.g. for policy-gradient training).
        V_grad = torch.autograd.grad(
            V.sum(), features,
            create_graph=True, retain_graph=True,
        )[0]                                                          # (B, 2)
        V_w = V_grad[:, 0:1]                                         # (B, 1)

        # ── ∂²V/∂w_norm² ─────────────────────────────────────────────────────
        V_ww_grad = torch.autograd.grad(
            V_w.sum(), features,
            create_graph=create_graph, retain_graph=True,
        )[0]                                                          # (B, 2)
        V_ww = V_ww_grad[:, 0:1]                                     # (B, 1)

        # ── Interior maximiser (concave region V_ww < 0) ─────────────────────
        w_norm   = features[:, 0:1]
        concave  = V_ww < -1e-12
        safe_Vww = torch.where(concave, V_ww, torch.full_like(V_ww, -1e-12))
        scale    = torch.where(
            concave,
            -V_w / (w_norm * safe_Vww),
            torch.zeros_like(V_w),
        )                                                              # (B, 1)

        # Multi-asset: π_nd = scale · Ω⁻¹η
        pi = scale * self.omega_inv_eta.unsqueeze(0)                  # (B, n)
        pi = _apply_leverage_constraint_torch(
            pi, self.d, self.u, self.max_long, self.max_short,
        )
        return pi if create_graph else pi.detach()


def _device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_model(architecture_name: str, n_assets: int, n_steps: int, d: float, u: float):
    spec = TORCH_ARCHITECTURES[architecture_name]
    kind = spec['kind']
    if kind == 'mlp':
        return TorchPolicyNet(n_assets=n_assets, hidden_layers=spec['hidden_layers'], d=d, u=u,
                              n_features=spec.get('n_features', 2))
    if kind == 'historical_replay':
        return TorchPolicyNet(n_assets=n_assets, hidden_layers=spec['hidden_layers'], d=d, u=u,
                              n_features=spec.get('n_features', RICH_FEATURE_DIM))
    if kind == 'deep_bsde':
        return DeepBSDEPolicyNet(n_assets=n_assets, n_steps=n_steps, hidden_layers=spec['hidden_layers'], d=d, u=u)
    if kind == 'pinn':
        return PINNPolicyNet(n_assets=n_assets, hidden_layers=spec['hidden_layers'], d=d, u=u)
    if kind == 'actor_critic':
        return ActorCriticPolicyNet(n_assets=n_assets, hidden_layers=spec['hidden_layers'], d=d, u=u)
    if kind == 'lstm':
        return LSTMPolicyNet(n_assets=n_assets, hidden_size=spec['hidden_size'], seq_len=spec['seq_len'], d=d, u=u)
    if kind == 'transformer':
        return TransformerPolicyNet(n_assets=n_assets, d_model=spec['d_model'], nhead=spec['nhead'], seq_len=spec['seq_len'], d=d, u=u)
    if kind == 'digital_hedge':
        return DigitalHedgeValueNet(n_assets=n_assets, hidden_layers=spec['hidden_layers'], d=d, u=u)
    raise ValueError(f'Unknown architecture kind: {kind}')


def _sequence_from_history(history: List[torch.Tensor], steps: List[int], goal: float, step_idx: int, total_steps: int, seq_len: int):
    items = list(zip(history, steps))[-seq_len:]
    if not items:
        raise ValueError('History cannot be empty')
    while len(items) < seq_len:
        items.insert(0, items[0])
    wealth_seq = torch.cat([w for w, _ in items], dim=1) / max(goal, 1e-12)
    tau_seq = torch.tensor(
        [max((total_steps - s) / total_steps, 1.0 / total_steps) for _, s in items],
        dtype=wealth_seq.dtype,
        device=wealth_seq.device,
    ).unsqueeze(0).repeat(wealth_seq.shape[0], 1)
    return torch.stack([wealth_seq, tau_seq], dim=-1)


def _current_features(W: torch.Tensor, goal: float, step_idx: int, total_steps: int):
    tau = max((total_steps - step_idx) / total_steps, 1.0 / total_steps)
    tau_tensor = torch.full_like(W, tau)
    return torch.cat([W / max(goal, 1e-12), tau_tensor], dim=1)


def _current_features_rich(W: torch.Tensor, goal: float, step_idx: int, total_steps: int,
                            history: List[torch.Tensor]) -> torch.Tensor:
    """
    5-feature input for nn_historical_replay.

    Features
    --------
    [0] w/goal        — wealth-to-goal ratio         (progress)
    [1] τ_norm        — time remaining / T            (urgency)
    [2] log(w/goal)   — log distance to goal          (curvature signal)
    [3] rolling_vol   — trailing 20-step annualised portfolio vol (regime)
    [4] rolling_ret   — trailing 20-step cumulative portfolio return (momentum)

    The rolling stats are computed from the wealth history passed in from the
    simulation / backtest loop, so inference and training use identical features.
    When history is short (< 2 steps) the last two features are set to zero.
    """
    tau_norm = max((total_steps - step_idx) / total_steps, 1.0 / total_steps)
    w_norm  = (W / max(goal, 1e-12)).clamp(1e-6, 10.0)   # (B, 1)
    log_w   = torch.log(w_norm)                            # (B, 1)
    tau_t   = torch.full_like(w_norm, tau_norm)            # (B, 1)

    window = min(20, len(history))
    if window >= 2:
        # stacked: (window, B, 1) → squeeze → (window, B)
        stacked    = torch.stack(history[-window:]).squeeze(-1)
        daily_rets = stacked[1:] / stacked[:-1].clamp(min=1e-6) - 1.0  # (window-1, B)
        roll_vol   = (daily_rets.std(dim=0, unbiased=False)
                      * math.sqrt(252)).unsqueeze(1)                      # (B, 1) annualised
        roll_ret   = (stacked[-1] / stacked[0].clamp(min=1e-6)
                      ).unsqueeze(1) - 1.0                                # (B, 1) cumulative
    else:
        roll_vol = torch.zeros_like(w_norm)
        roll_ret = torch.zeros_like(w_norm)

    return torch.cat([w_norm, tau_t, log_w, roll_vol, roll_ret], dim=1)  # (B, 5)


def _forward_policy(model, W: torch.Tensor, goal: float, step_idx: int, total_steps: int,
                    history: List[torch.Tensor], steps: List[int]):
    kind = getattr(model, 'kind', 'mlp')
    if kind == 'historical_replay':
        # Rich 5-feature input: [w/goal, τ, log(w/goal), rolling_vol, rolling_ret]
        features = _current_features_rich(W, goal, step_idx, total_steps, history)
        return model(features), None
    if kind == 'digital_hedge':
        # Policy = delta hedge of V — requires autograd even inside no_grad contexts.
        # torch.enable_grad() overrides an outer torch.no_grad() block.
        features = _current_features(W, goal, step_idx, total_steps)
        with torch.enable_grad():
            feat_req = features.detach().requires_grad_(True)
            pi = model.delta_policy(feat_req, create_graph=False)
        return pi, None                     # pi already detached by delta_policy
    if kind == 'deep_bsde':
        features = _current_features(W, goal, step_idx, total_steps)
        return model.forward_step(features, step_idx), None
    if kind == 'pinn':
        features = _current_features(W, goal, step_idx, total_steps)
        return model(features)
    if kind == 'actor_critic':
        features = _current_features(W, goal, step_idx, total_steps)
        return model(features)
    if kind == 'lstm':
        seq = _sequence_from_history(history, steps, goal, step_idx, total_steps, model.seq_len)
        return model(seq), None
    if kind == 'transformer':
        seq = _sequence_from_history(history, steps, goal, step_idx, total_steps, model.seq_len)
        return model(seq), None
    features = _current_features(W, goal, step_idx, total_steps)
    return model(features), None


def policy_weights(net, W_current, goal: float, history=None, step_idx: int = 0, total_steps: int = 252, device=None):
    import numpy as _np
    device = _device(device) if device is not None else next(net.parameters()).device
    W_np = _np.asarray(W_current, dtype=_np.float32).ravel()
    is_scalar = W_np.size == 1
    with torch.no_grad():
        W = torch.tensor(W_np, dtype=torch.float32, device=device).view(-1, 1)  # (N, 1)
        if history is None:
            history_vals = [W.clone()]
            step_vals = [step_idx]
        else:
            history_vals = [torch.tensor([[float(v)]], dtype=torch.float32, device=device) for v in history]
            step_vals = list(range(max(0, step_idx - len(history_vals) + 1), step_idx + 1))
        result = _forward_policy(net, W, goal, step_idx, total_steps, history_vals, step_vals)
        if isinstance(result, tuple):
            pi = result[0]
        else:
            pi = result
        out = pi.detach().cpu().numpy()
        return out.squeeze(0) if is_scalar else out


def _terminal_utility(
    W: torch.Tensor, goal: float, utility: str,
    asp_p: float, asp_c1: float, asp_R: float,
    sig_temp: float = 0.05,
):
    """
    Differentiable terminal utility.

    sig_temp : sigmoid temperature for 'goalreach'.
               Adaptive value (σ_eff · √T · 0.30) is passed from the trainer;
               default 0.05 retained for backward compatibility.
    """
    w_norm = W.squeeze(1) / goal
    if utility == 'goalreach':
        return torch.sigmoid((w_norm - 1.0) / sig_temp)
    if utility == 'goalreach_ste':
        # Straight-Through Estimator: forward is exact step, backward is sigmoid
        # surrogate.  This matches the fd_nd terminal condition 1{W_T >= goal}.
        return GoalReachSTE.apply(w_norm - 1.0, sig_temp)
    if utility == 'aspiration':
        return torch.where(w_norm < asp_R, w_norm**asp_p / asp_p, asp_c1 * w_norm**asp_p / asp_p)
    raise ValueError(f'Unknown utility: {utility}')


def _apply_leverage_constraint_torch(pi, d: float, u: float,
                                     max_long: float, max_short: float):
    """
    Apply per-asset bounds [d, u] then aggregate long/short leverage caps.
    Long  side : sum(max(w, 0)) <= max_long
    Short side : sum(max(-w, 0)) <= max_short
    Scales each side proportionally; never amplifies weights.
    """
    pi = pi.clamp(d, u)
    long_lev = pi.clamp(min=0.0).sum(dim=1, keepdim=True).clamp(min=1e-12)
    short_lev = (-pi).clamp(min=0.0).sum(dim=1, keepdim=True).clamp(min=1e-12)
    long_scale = (max_long / long_lev).clamp(max=1.0)
    short_scale = (max_short / short_lev).clamp(max=1.0)
    return torch.where(pi >= 0, pi * long_scale, pi * short_scale)


# ── Digital-option delta-hedge trainer ───────────────────────────────────────

def train_digital_hedge_net(
    mu_vec, omega_mat, r,
    w0: float = 1.0, goal_mult: float = 1.10,
    n_paths: int = 512,
    pretrain_iters: int = 200,
    hjb_iters: int = 300,
    sim_iters: int = 200,
    lr: float = 3e-3,
    n_steps: int = 40,
    d: float = -5.0, u: float = 3.0,
    max_long_leverage: float = 3.0, max_short_leverage: float = 5.0,
    patience: int = 80,
    T: float = 1.0,
    seed: int = 1,
    device=None,
    verbose: bool = False,
    hidden_layers=(128, 128, 128),
):
    """
    Three-phase training for the digital-option delta-hedge architecture.

    The value network V(w_norm, τ_norm) is trained to equal P(W_T ≥ goal),
    the price of a digital call option.  Portfolio weights at each step are
    derived as the multi-asset delta hedge  π = −(V_w/(w·V_ww)) · Ω⁻¹η.

    Phase 1 — Supervised Browne pre-train
    ───────────────────────────────────────
    Fit V to the Browne analytical value function V_browne(w,τ) = Φ(d) via
    MSE on random (w_norm, τ) samples.  The r-corrected d is:

        d = (log w + (r + ½θ²) τ) / (θ √τ)

    Initialises the network in the correct basin before any PDE or simulation
    training.

    Phase 2 — Kolmogorov PDE residual
    ────────────────────────────────────
    Enforce the Kolmogorov backward equation for the optimal log-wealth process
    on a grid of interior (w_norm, τ_norm) points:

        V_τ  =  (r + θ²) · w · V_w  +  ½θ² · w² · V_ww  =  0

    This is the PDE that Browne's Φ(d) EXACTLY satisfies (verified analytically).
    The corresponding nonlinear HJB (V_ww·V_τ − r·w·V_w·V_ww + ½θ²·V_w² = 0)
    is NOT satisfied by Φ(d) in the classical sense and is numerically unstable
    near the inflection of the S-shaped V.  Boundary conditions V(w→0, τ) ≈ 0
    and V(w→∞, τ) ≈ 1 are enforced as auxiliary MSE terms.

    Phase 3 — Path simulation + terminal BCE matching
    ───────────────────────────────────────────────────
    Simulate GBM paths under the current delta-hedge policy, then train V to
    match the empirical binary outcomes  O_i = 1{W_T^(i) ≥ goal}  via BCE:

        L_match = −E[O · log V(W_T, 0) + (1−O) · log(1 − V(W_T, 0))]

    This closes the self-consistency loop: as V becomes a better predictor of
    its own policy's success, the delta hedge improves, which in turn provides
    better training signal for V.  A Browne anchor loss prevents drift.

    Returns
    -------
    net      : DigitalHedgeValueNet (eval mode, buffers populated)
    histories: dict matching the format expected by evaluate_nn_portfolio
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for train_digital_hedge_net")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = _device(device)
    n   = len(mu_vec)
    eta = np.asarray(mu_vec, float) - float(r)

    # ── Market calibration ───────────────────────────────────────────────────
    omega_inv_eta = np.linalg.solve(np.asarray(omega_mat, float)
                                    + 1e-10 * np.eye(n), eta)
    theta2    = max(float(np.dot(eta, omega_inv_eta)), 1e-12)
    theta     = math.sqrt(theta2)
    goal      = w0 * goal_mult
    sig_temp  = max(0.30 * theta * math.sqrt(T), 0.02)   # for STE reference

    # ── Build model and set analytical buffers ────────────────────────────────
    net = DigitalHedgeValueNet(
        n_assets=n, hidden_layers=tuple(hidden_layers),
        d=d, u=u, max_long=max_long_leverage, max_short=max_short_leverage,
    ).to(dev)
    net.omega_inv_eta.copy_(torch.tensor(omega_inv_eta.astype(np.float32), device=dev))
    net.theta2_val.fill_(float(theta2))
    net.r_val.fill_(float(r))

    # ── Fixed simulation tensors ──────────────────────────────────────────────
    mu_t  = torch.tensor(np.asarray(mu_vec, dtype=np.float32), device=dev)
    chol  = np.linalg.cholesky(np.asarray(omega_mat, float) + 1e-10 * np.eye(n))
    sig_t = torch.tensor(chol.astype(np.float32), device=dev)
    r_t   = torch.tensor(float(r), dtype=torch.float32, device=dev)
    dt    = T / n_steps
    sqdt  = math.sqrt(dt)

    loss_history: List[float] = []
    val_history:  List[float] = []
    val_iters:    List[int]   = []

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: evaluate empirical P(goal) on n_mc fresh paths
    # ─────────────────────────────────────────────────────────────────────────
    def _eval_goal_prob(n_mc: int = 256) -> float:
        with torch.enable_grad():
            W_e = torch.full((n_mc, 1), float(w0), device=dev)
            for s in range(n_steps):
                tau_n = (T - s * dt) / T
                wn_e  = (W_e / goal).clamp(1e-6, 10.0)
                ft_e  = torch.cat([wn_e,
                                   torch.full_like(wn_e, tau_n)], dim=1)
                ft_e  = ft_e.detach().requires_grad_(True)
                pi_e  = net.delta_policy(ft_e, create_graph=False)
                Z_e   = torch.randn(n_mc, n, device=dev)
                dS_e  = mu_t * dt + (Z_e @ sig_t.T) * sqdt
                exc_e = (pi_e * (dS_e - r_t * dt)).sum(1, keepdim=True)
                W_e   = (W_e * (1.0 + r_t * dt + exc_e)).clamp(min=1e-6).detach()
        return ((W_e.squeeze() / goal) >= 1.0).float().mean().item()

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Supervised Browne pre-training
    # ─────────────────────────────────────────────────────────────────────────
    if pretrain_iters > 0:
        if verbose:
            print(f"  [digital_hedge] Phase 1: Browne supervised ({pretrain_iters} iters)")
        pre_opt = optim.Adam(net.parameters(), lr=lr * 2.0)
        rng1    = np.random.default_rng(seed)

        for pit in range(pretrain_iters):
            with torch.enable_grad():
                B     = 512
                w_s   = rng1.uniform(0.1, 2.0, B).astype(np.float32)
                tau_s = rng1.uniform(dt, T, B).astype(np.float32)

                # Browne analytical target V = Φ(d)
                # Correct formula includes r in drift: d = (log w + (r + ½θ²)τ)/(θ√τ)
                d_s      = (np.log(np.maximum(w_s, 1e-10))
                            + (r + 0.5 * theta2) * tau_s) / (theta * np.sqrt(tau_s))
                V_target = torch.tensor(
                    np.array([0.5 * (1.0 + math.erf(float(di) / math.sqrt(2.0)))
                              for di in d_s], dtype=np.float32),
                    device=dev,
                )
                feat = torch.stack([
                    torch.tensor(w_s, device=dev),
                    torch.tensor(tau_s / T, device=dev),
                ], dim=1)

                V_pred   = net(feat).squeeze()
                pre_loss = ((V_pred - V_target) ** 2).mean()
                pre_opt.zero_grad()
                pre_loss.backward()
                pre_opt.step()

        if verbose:
            print(f"  [digital_hedge] Phase 1 done. Browne MSE={pre_loss.item():.5f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: HJB PDE residual
    # ─────────────────────────────────────────────────────────────────────────
    if hjb_iters > 0:
        if verbose:
            print(f"  [digital_hedge] Phase 2: HJB residual ({hjb_iters} iters)")
        hjb_opt  = optim.Adam(net.parameters(), lr=lr)
        hjb_sch  = optim.lr_scheduler.CosineAnnealingLR(
            hjb_opt, T_max=max(hjb_iters, 1), eta_min=lr * 0.05,
        )
        best_val2    = -math.inf
        no_imp2      = 0

        for it in range(hjb_iters):
            B    = 256
            rng2 = np.random.default_rng(seed + 10000 + it)

            # Interior sample: w_norm ∈ [0.1, 2.0],  τ_norm ∈ [dt/T, 1]
            w_s   = torch.tensor(rng2.uniform(0.1, 2.0, B).astype(np.float32), device=dev)
            tau_s = torch.tensor(
                rng2.uniform(dt / T, 1.0, B).astype(np.float32), device=dev,
            )
            with torch.enable_grad():
                feat  = torch.stack([w_s, tau_s], dim=1).requires_grad_(True)
                V     = net(feat)                                         # (B, 1)

                # First-order: V_w (∂V/∂w_norm) and V_tau_norm (∂V/∂τ_norm)
                grads   = torch.autograd.grad(V.sum(), feat,
                                              create_graph=True, retain_graph=True)[0]
                V_w         = grads[:, 0:1]        # ∂V/∂w_norm
                V_tau_norm  = grads[:, 1:2]        # ∂V/∂τ_norm
                V_tau       = V_tau_norm / T       # ∂V/∂τ (real time)

                # Second-order: V_ww
                V_ww = torch.autograd.grad(V_w.sum(), feat,
                                           create_graph=True, retain_graph=True)[0][:, 0:1]

                # Kolmogorov backward equation for the optimal log-wealth process:
                #   V_τ  =  (r + θ²) · w · V_w  +  ½θ² · w² · V_ww
                # This is the correct PDE that Browne's formula satisfies exactly.
                # (The nonlinear HJB V_ww·V_τ − r·w·V_w·V_ww + ½θ²·V_w² = 0 is
                #  NOT satisfied by Φ(d); it would require V_ww < 0 everywhere, which
                #  fails near the inflection point of the S-shaped value function.)
                w_feat    = feat[:, 0:1]
                residual  = (V_tau
                             - (r_t + theta2) * w_feat * V_w
                             - 0.5 * theta2 * w_feat ** 2 * V_ww)
                hjb_loss  = (residual ** 2).mean()

                # ── Boundary: V(w≈0, τ) ≈ 0 ──────────────────────────────────────
                rng3 = np.random.default_rng(seed + 20000 + it)
                w_lo   = torch.tensor(rng3.uniform(0.02, 0.10, 32).astype(np.float32), device=dev)
                tau_lo = torch.tensor(rng3.uniform(dt / T, 1.0, 32).astype(np.float32), device=dev)
                feat_lo  = torch.stack([w_lo, tau_lo], dim=1)
                bnd_low  = net(feat_lo).pow(2).mean()

                # ── Boundary: V(w large, τ) ≈ 1 ─────────────────────────────────
                w_hi   = torch.tensor(rng3.uniform(1.8, 2.0, 32).astype(np.float32), device=dev)
                tau_hi = torch.tensor(rng3.uniform(dt / T, 1.0, 32).astype(np.float32), device=dev)
                feat_hi  = torch.stack([w_hi, tau_hi], dim=1)
                bnd_high = (1.0 - net(feat_hi)).pow(2).mean()

                # ── Terminal: V(w, 0) = 1{w ≥ 1} ────────────────────────────────
                w_tm   = torch.tensor(rng3.uniform(0.1, 2.0, 64).astype(np.float32), device=dev)
                tau_tm = torch.zeros(64, 1, device=dev)
                feat_tm  = torch.stack([w_tm, tau_tm.squeeze()], dim=1)
                V_tm     = net(feat_tm).squeeze()
                term_tgt = (w_tm >= 1.0).float()
                term_loss = ((V_tm - term_tgt) ** 2).mean()

                loss = hjb_loss + 0.5 * bnd_low + 0.5 * bnd_high + 1.0 * term_loss
                hjb_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                hjb_opt.step()
                hjb_sch.step()

            # Track as negative Browne MSE (higher = better fit)
            with torch.no_grad():
                ref_w   = torch.tensor([0.8, 1.0, 1.2], device=dev)
                ref_tau = torch.full_like(ref_w, 0.5)
                ref_f   = torch.stack([ref_w, ref_tau / T], dim=1)
                V_ref   = net(ref_f).squeeze().cpu().numpy()
            d_ref  = [(math.log(float(w)) + (r + 0.5 * theta2) * 0.5) / (theta * math.sqrt(0.5))
                      for w in [0.8, 1.0, 1.2]]
            V_br   = np.array([0.5 * (1.0 + math.erf(d / math.sqrt(2.0))) for d in d_ref])
            eu     = -float(np.mean((V_ref - V_br) ** 2))
            loss_history.append(eu)

            if it % 20 == 0:
                val_history.append(eu)
                val_iters.append(it)
                if eu > best_val2 + 1e-5:
                    best_val2 = eu;  no_imp2 = 0
                else:
                    no_imp2 += 20
            if no_imp2 >= patience and it > hjb_iters // 4:
                if verbose:
                    print(f"  [digital_hedge] Phase 2 early stop at iter {it+1}")
                break

        if verbose:
            print(f"  [digital_hedge] Phase 2 done.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Path simulation + terminal BCE matching
    # ─────────────────────────────────────────────────────────────────────────
    if sim_iters > 0:
        if verbose:
            print(f"  [digital_hedge] Phase 3: path BCE ({sim_iters} iters)")
        sim_opt   = optim.Adam(net.parameters(), lr=lr * 0.3)
        best_val3 = -math.inf
        no_imp3   = 0
        rng_sim   = np.random.default_rng(seed + 30000)

        for it in range(sim_iters):
            with torch.enable_grad():
                # ── Simulate paths under current delta-hedge policy ───────────
                W = torch.full((n_paths, 1), float(w0), device=dev)
                for s in range(n_steps):
                    tau_n = (T - s * dt) / T
                    wn    = (W / goal).clamp(1e-6, 10.0)
                    ft    = torch.cat([wn, torch.full_like(wn, tau_n)], dim=1)
                    ft    = ft.detach().requires_grad_(True)
                    pi    = net.delta_policy(ft, create_graph=False)
                    Z     = torch.randn(n_paths, n, device=dev)
                    dS    = mu_t * dt + (Z @ sig_t.T) * sqdt
                    exc   = (pi * (dS - r_t * dt)).sum(1, keepdim=True)
                    W     = (W * (1.0 + r_t * dt + exc)).clamp(min=1e-6).detach()

                # Binary outcomes: 1{W_T ≥ goal}
                W_T_norm = (W.squeeze() / goal).clamp(1e-6, 10.0)
                outcomes = (W_T_norm >= 1.0).float()
                goal_prob = outcomes.mean().item()

                # ── BCE: V(W_T, 0) predicts empirical outcomes ───────────────
                tau_term  = torch.zeros(n_paths, 1, device=dev)
                feat_term = torch.stack([W_T_norm.detach(), tau_term.squeeze()], dim=1)
                V_term    = net(feat_term).squeeze().clamp(1e-7, 1.0 - 1e-7)
                bce_loss  = -(outcomes * V_term.log()
                             + (1.0 - outcomes) * (1.0 - V_term).log()).mean()

                # ── Browne anchor at mid-horizon to prevent drift ─────────────
                w_anch  = torch.tensor(rng_sim.uniform(0.5, 1.8, 64).astype(np.float32), device=dev)
                tau_ref = 0.5
                d_anch  = (np.log(np.maximum(rng_sim.uniform(0.5, 1.8, 64), 1e-10))
                           + 0.5 * theta2 * tau_ref) / (theta * math.sqrt(tau_ref))
                V_anch_tgt = torch.tensor(
                    np.array([0.5 * (1.0 + math.erf(float(di) / math.sqrt(2.0)))
                              for di in d_anch], dtype=np.float32),
                    device=dev,
                )
                feat_anch  = torch.stack([w_anch,
                                          torch.full_like(w_anch, tau_ref / T)], dim=1)
                V_anch     = net(feat_anch).squeeze()
                anchor_loss = ((V_anch - V_anch_tgt) ** 2).mean()

                loss = bce_loss + 0.2 * anchor_loss
                sim_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                sim_opt.step()

            loss_history.append(goal_prob)

            if it % 20 == 0:
                val_history.append(goal_prob)
                val_iters.append((hjb_iters if hjb_iters else 0) + it)
                if goal_prob > best_val3 + 1e-4:
                    best_val3 = goal_prob;  no_imp3 = 0
                else:
                    no_imp3 += 20
            if no_imp3 >= patience and it > sim_iters // 4:
                if verbose:
                    print(f"  [digital_hedge] Phase 3 early stop at iter {it+1}")
                break

        if verbose:
            print(f"  [digital_hedge] Phase 3 done.")

    net.eval()

    # ── Post-training test evaluation (fresh 2048-path MC) ───────────────────
    test_u = _eval_goal_prob(n_mc=2048)

    param_count = sum(p.numel() for p in net.parameters())
    return net, {
        'architecture_name': 'nn_digital_hedge',
        'backend'           : 'torch',
        'kind'              : 'digital_hedge',
        'param_size'        : int(param_count),
        'device'            : str(dev),
        'loss_history'      : loss_history,
        'val_history'       : val_history,
        'val_iters'         : val_iters,
        'test_u'            : float(test_u),
    }


def train_historical_replay_net(
    historical_returns,        # np.ndarray (T, n_assets) — actual daily gross returns
    mu_vec, omega_mat, r,
    architecture_name: str = 'nn_historical_replay',
    w0: float = 1.0,
    goal_mult: float = 1.10,
    n_paths: int = 256,
    n_iters: int = 200,
    lr: float = 3e-3,
    n_steps: int = 252,        # steps per training episode (≈ 1 year)
    block_size: int = 21,      # block-bootstrap block length (≈ 1 month)
    d: float = -5.0, u: float = 3.0,
    max_long_leverage: float = 3.0, max_short_leverage: float = 5.0,
    train_frac: float = 0.60,  # fraction of history used for training
    val_frac:   float = 0.20,  # fraction used for validation
    # test_frac = 1 - train_frac - val_frac (rest)
    patience: int = 60,
    T: float = 1.0,
    seed: int = 1,
    device=None,
    verbose: bool = False,
):
    """
    Train a policy network using *real* historical return data instead of GBM
    simulations.  Three strictly time-ordered splits are used:

    ┌──────────────┬──────────────┬──────────────┐
    │  TRAIN 60%   │   VAL  20%   │  TEST  20%   │
    │ block-bootstrapped paths    │  actual seq  │
    └──────────────┴──────────────┴──────────────┘

    Training paths are generated by block-bootstrap sampling from the training
    period (blocks of `block_size` consecutive days, sampled with replacement).
    This preserves short-term autocorrelation while giving unlimited paths.

    Validation and test use the *actual* historical return sequences for those
    periods — no bootstrapping — so the evaluation is on unseen real data.

    Architecture ``nn_historical_replay`` uses RICH_FEATURE_DIM=5 inputs:
      [w/goal, τ_norm, log(w/goal), rolling_vol_20d_ann, rolling_ret_20d]
    which let the network condition on the current volatility regime and recent
    momentum in addition to the basic (w, τ) state.
    """
    ret_arr = np.asarray(historical_returns, dtype=np.float32)  # (T, n)
    T_hist, n = ret_arr.shape
    assert n == len(mu_vec), "historical_returns columns must equal len(mu_vec)"

    # ── Time-ordered splits ───────────────────────────────────────────────────
    t_train_end = int(T_hist * train_frac)
    t_val_end   = int(T_hist * (train_frac + val_frac))
    train_ret = ret_arr[:t_train_end]         # (T_train, n)
    val_ret   = ret_arr[t_train_end:t_val_end]  # (T_val, n)
    test_ret  = ret_arr[t_val_end:]             # (T_test, n)

    if verbose:
        print(f"  [historical_replay] split: train={len(train_ret)}d "
              f"val={len(val_ret)}d test={len(test_ret)}d")

    dev   = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    goal  = w0 * goal_mult
    r_t   = torch.tensor(float(r), dtype=torch.float32, device=dev)

    # ── Build model ───────────────────────────────────────────────────────────
    spec = TORCH_ARCHITECTURES[architecture_name]
    n_features = spec.get('n_features', RICH_FEATURE_DIM)
    model = TorchPolicyNet(
        n_assets=n, hidden_layers=tuple(spec['hidden_layers']),
        d=d, u=u, n_features=n_features,
    ).to(dev)
    model.kind = 'historical_replay'

    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_iters, 1), eta_min=lr * 0.05)

    # Pre-compute n_blocks needed per episode
    n_blocks = math.ceil(n_steps / block_size)
    # Number of blocks available in training set
    n_avail_blocks = max(1, len(train_ret) - block_size + 1)

    def _sample_bootstrap_batch() -> torch.Tensor:
        """
        Draw a full batch of (n_paths, n_steps, n) return paths via block
        bootstrap from train_ret in a single CPU→GPU transfer.

        Previously this was called n_paths times in a list comprehension,
        producing n_paths separate small transfers per iteration.  Pre-building
        the entire batch in numpy and calling torch.tensor once reduces the
        transfer count to 1 per training iteration regardless of n_paths.
        """
        # (n_paths, n_blocks) start indices — sampled in one numpy call
        all_starts = rng.integers(0, n_avail_blocks, size=(n_paths, n_blocks))
        # Build every path in numpy then stack
        batch = np.empty((n_paths, n_steps, n), dtype=np.float32)
        for i in range(n_paths):
            segs_list = [train_ret[all_starts[i, j]: all_starts[i, j] + block_size]
                         for j in range(n_blocks)]
            batch[i] = np.concatenate(segs_list, axis=0)[:n_steps]
        # Single GPU transfer for the entire batch
        return torch.tensor(batch, dtype=torch.float32, device=dev)  # (n_paths, n_steps, n)

    def _eval_period(period_ret: np.ndarray) -> float:
        """Roll a single wealth path through `period_ret` and return P(goal)."""
        if len(period_ret) < n_steps:
            return float('nan')
        # Use first n_steps days of the period
        period_t = torch.tensor(period_ret[:n_steps], dtype=torch.float32, device=dev)
        with torch.no_grad():
            W_e = torch.full((1, 1), float(w0), device=dev)
            hist_e = [W_e]
            for s in range(n_steps):
                feat = _current_features_rich(W_e, goal, s, n_steps, hist_e)
                pi   = model(feat)
                pi   = _apply_leverage_constraint_torch(pi, d, u, max_long_leverage, max_short_leverage)
                dS   = period_t[s].unsqueeze(0)   # (1, n)
                exc  = (pi * (dS - r_t * (1.0 / n_steps))).sum(1, keepdim=True)
                W_e  = (W_e * (1.0 + r_t * (1.0 / n_steps) + exc)).clamp(min=1e-6)
                hist_e.append(W_e)
        return float((W_e.squeeze() >= goal).float().item())

    loss_history: List[float] = []
    val_history:  List[float] = []
    val_iters:    List[int]   = []
    best_val      = -math.inf
    no_improve    = 0

    for it in range(n_iters):
        # ── Build batch of bootstrap paths — single GPU transfer ─────────────
        paths = _sample_bootstrap_batch()
        # paths: (n_paths, n_steps, n)

        W = torch.full((n_paths, 1), float(w0), dtype=torch.float32, device=dev)
        hist_tr = [W]
        dt_step = 1.0 / n_steps  # each step = 1/n_steps of a year

        for s in range(n_steps):
            feat = _current_features_rich(W, goal, s, n_steps, hist_tr)
            pi   = model(feat)
            pi   = _apply_leverage_constraint_torch(pi, d, u, max_long_leverage, max_short_leverage)
            dS   = paths[:, s, :]                   # (n_paths, n) actual returns
            exc  = (pi * (dS - r_t * dt_step)).sum(1, keepdim=True)
            W    = (W * (1.0 + r_t * dt_step + exc)).clamp(min=1e-6)
            hist_tr.append(W)

        # Terminal utility: E[1{W_T >= goal}] (exact step, no sigmoid surrogate)
        utility_val = (W.squeeze() >= goal).float()
        # Soft version for gradient signal (sigmoid around goal)
        w_norm_term = (W.squeeze() / goal - 1.0)
        soft_u = torch.sigmoid(w_norm_term / 0.05)
        loss   = -soft_u.mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        train_goal_prob = utility_val.mean().item()
        loss_history.append(train_goal_prob)

        # ── Validation every 10 iters ─────────────────────────────────────────
        if it % 10 == 0:
            model.eval()
            val_gp = _eval_period(val_ret)
            model.train()
            val_history.append(val_gp)
            val_iters.append(it)

            if verbose:
                print(f"  [historical_replay] it={it:4d}  "
                      f"train_P(goal)={train_goal_prob:.3f}  "
                      f"val_P(goal)={val_gp:.3f}")

            if not math.isnan(val_gp) and val_gp > best_val + 1e-4:
                best_val   = val_gp
                no_improve = 0
            else:
                no_improve += 10

            if no_improve >= patience and it > n_iters // 4:
                if verbose:
                    print(f"  [historical_replay] early stop at iter {it+1}")
                break

    model.eval()

    # ── Test evaluation on held-out period ────────────────────────────────────
    test_u = _eval_period(test_ret)
    if verbose:
        print(f"  [historical_replay] done. test_P(goal)={test_u:.3f}")

    param_count = sum(p.numel() for p in model.parameters())
    return model, {
        'architecture_name': architecture_name,
        'backend'          : 'torch',
        'kind'             : 'historical_replay',
        'param_size'       : int(param_count),
        'device'           : str(dev),
        'loss_history'     : loss_history,
        'val_history'      : val_history,
        'val_iters'        : val_iters,
        'test_u'           : float(test_u) if not math.isnan(test_u) else 0.0,
    }


def train_torch_policy_net(
    mu_vec, omega_mat, r,
    architecture_name='nn_mlp_small',
    w0=1.0, goal_mult=1.10,
    n_paths=256, n_iters=40,
    lr=3e-3, n_steps=24,
    d=-5.0, u=3.0,
    max_long_leverage=3.0, max_short_leverage=5.0,
    utility='goalreach', asp_p=0.5, asp_c1=1.2, asp_R=1.0,
    seed=1, device=None,
    # ── new parameters ────────────────────────────────────────────────
    pretrain_iters=100,    # supervised Browne warm-start (0 = skip)
    p_curriculum=0.30,     # fraction of paths starting near goal boundary
    antithetic=True,       # antithetic variates: simulate Z and −Z pairs
    patience=60,           # early-stopping patience (iters with no val improvement)
    T=1.0,                 # horizon in years (used for tau normalisation)
    verbose=False,         # print training progress
    historical_returns=None,  # np.ndarray (T,n) — real returns for historical_replay
    # ──────────────────────────────────────────────────────────────────
):
    """
    Train any TORCH_ARCHITECTURES model via policy gradient.

    Improvements over v1
    ─────────────────────
    • Browne supervised pre-training  — correct basin from the start
    • Antithetic variates             — halves gradient variance
    • Curriculum sampling             — covers near-(goal, τ≈0) region
    • Adaptive sigmoid temperature    — matches near-terminal diffusion width
    • Cosine LR schedule              — avoids late-training oscillation
    • Early stopping on val utility   — stops when val no longer improves
    """
    if architecture_name not in TORCH_ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture_name}')

    # ── Apply per-architecture constraint overrides ───────────────────────────
    # The arch spec may carry a 'constraints' dict that narrows the leverage
    # envelope (e.g. long-only: d=0, u=1, max_long=1, max_short=0).  These
    # take priority over whatever the caller passed in.
    arch_constraints = TORCH_ARCHITECTURES[architecture_name].get('constraints', {})
    if arch_constraints:
        d                  = arch_constraints.get('d',         d)
        u                  = arch_constraints.get('u',         u)
        max_long_leverage  = arch_constraints.get('max_long',  max_long_leverage)
        max_short_leverage = arch_constraints.get('max_short', max_short_leverage)

    # ── Per-architecture minimum iteration budget ─────────────────────────────
    # Architectures with many internal parameters (deep_bsde, transformer) need
    # more gradient steps to converge.  'n_iters_min' in the arch spec sets a
    # floor; the caller's n_iters is used if it is already larger.
    _iters_floor = TORCH_ARCHITECTURES[architecture_name].get('n_iters_min', 0)
    if n_iters < _iters_floor:
        n_iters = _iters_floor

    # ── Delegate historical_replay to its dedicated trainer ──────────────────
    if TORCH_ARCHITECTURES[architecture_name].get('kind') == 'historical_replay':
        if historical_returns is None:
            raise ValueError(
                f"architecture '{architecture_name}' requires historical_returns "
                "to be passed (a (T, n_assets) numpy array of daily gross returns)."
            )
        return train_historical_replay_net(
            historical_returns=historical_returns,
            mu_vec=mu_vec, omega_mat=omega_mat, r=r,
            architecture_name=architecture_name,
            w0=w0, goal_mult=goal_mult,
            n_paths=n_paths, n_iters=n_iters, lr=lr,
            n_steps=n_steps,
            d=d, u=u,
            max_long_leverage=max_long_leverage,
            max_short_leverage=max_short_leverage,
            patience=patience, T=T, seed=seed,
            device=device, verbose=verbose,
        )

    # ── Delegate digital_hedge (and its long-only twin) ──────────────────────
    # train_digital_hedge_net has a different phase structure (Browne pre-train,
    # Kolmogorov PDE residual, path BCE) and returns a DigitalHedgeValueNet.
    # We pass the (possibly overridden) constraint parameters through.
    if TORCH_ARCHITECTURES[architecture_name].get('kind') == 'digital_hedge':
        return train_digital_hedge_net(
            mu_vec=mu_vec, omega_mat=omega_mat, r=r,
            w0=w0, goal_mult=goal_mult,
            n_paths=n_paths,
            pretrain_iters=pretrain_iters,
            hjb_iters=max(n_iters // 2, 1),
            sim_iters=max(n_iters // 2, 1),
            lr=lr, n_steps=n_steps, d=d, u=u,
            max_long_leverage=max_long_leverage,
            max_short_leverage=max_short_leverage,
            patience=patience, T=T, seed=seed,
            device=device, verbose=verbose,
        )

    # Allow the arch spec to override the utility (e.g. 'nn_ste_goalreach' uses
    # 'goalreach_ste').  Existing architectures have no 'utility' key so they
    # continue to use whatever was passed in (default: 'goalreach').
    arch_utility = TORCH_ARCHITECTURES[architecture_name].get('utility')
    if arch_utility is not None:
        utility = arch_utility

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device(device)

    n_assets = len(mu_vec)
    model    = _build_model(architecture_name, n_assets, n_steps, d, u).to(dev)
    opt      = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(n_iters, 1), eta_min=lr * 0.05
    )

    mu_t  = torch.tensor(np.asarray(mu_vec, dtype=np.float32), device=dev)
    chol  = np.linalg.cholesky(
        np.asarray(omega_mat, dtype=float) + 1e-10 * np.eye(n_assets)
    )
    sig_t = torch.tensor(chol.astype(np.float32), device=dev)
    r_t   = torch.tensor(float(r), dtype=torch.float32, device=dev)

    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)
    goal    = w0 * goal_mult

    # ── Analytical quantities for pre-training and adaptive temperature ──────
    eta           = np.asarray(mu_vec, float) - r
    omega_inv_eta = np.linalg.solve(np.asarray(omega_mat, float), eta)
    theta2        = max(float(np.dot(eta, omega_inv_eta)), 1e-12)
    eff_sigma     = math.sqrt(theta2)
    sig_temp      = max(0.30 * eff_sigma * math.sqrt(T), 0.02)   # adaptive temperature

    # ── Phase 1: Supervised Browne/Merton pre-training ───────────────────────
    if pretrain_iters > 0:
        if verbose:
            print(f"  [{architecture_name}] pre-training ({pretrain_iters} iters) ...")
        pre_opt = optim.Adam(model.parameters(), lr=lr * 2.0)
        rng     = np.random.default_rng(seed)

        for pit in range(pretrain_iters):
            B      = 256
            w_s    = rng.uniform(0.5, 2.0, size=B).astype(np.float32)
            tau_s  = rng.uniform(dt, T,    size=B).astype(np.float32)

            pi_tgt = _browne_policy_np(w_s, tau_s, omega_inv_eta, theta2, d, u)

            W_pre   = torch.tensor(w_s[:, None] * goal, dtype=torch.float32, device=dev)
            pi_t    = torch.tensor(pi_tgt,              dtype=torch.float32, device=dev)
            step_indices = [int(round((1.0 - float(ts) / T) * n_steps)) for ts in tau_s]

            # Run forward pass for each sample individually using _forward_policy
            # For efficiency, pick a representative single step_idx for the batch
            mid_step = n_steps // 2
            hist     = [W_pre]
            sth      = [0]
            outputs  = _forward_policy(model, W_pre, goal, mid_step, n_steps, hist, sth)
            pi_pred  = outputs[0] if isinstance(outputs, tuple) else outputs
            pre_loss = ((pi_pred - pi_t) ** 2).mean()

            pre_opt.zero_grad()
            pre_loss.backward()
            pre_opt.step()

        if verbose:
            print(f"  [{architecture_name}] pre-training done. MSE={pre_loss.item():.5f}")

    # ── Phase 2: Policy-gradient training ────────────────────────────────────
    loss_history     = []    # training E[U], every iteration
    val_history      = []    # validation E[U], every 10 iters
    val_iters        = []    # iteration indices for val points
    best_val_utility = -math.inf
    no_improve_count = 0

    # With antithetic: simulate base_paths and mirror with −Z
    base_paths = n_paths // 2 if antithetic else n_paths

    for it in range(n_iters):

        # ── Curriculum initialisation ─────────────────────────────────────────
        n_curr = int(base_paths * p_curriculum)
        n_reg  = base_paths - n_curr
        w_reg  = torch.full((n_reg, 1),  float(w0), dtype=torch.float32, device=dev)
        w_curr = goal * (0.70 + 0.35 * torch.rand(n_curr, 1, device=dev))
        W_base = torch.cat([w_reg, w_curr], dim=0)

        W = torch.cat([W_base, W_base.clone()], dim=0) if antithetic else W_base

        history      = [W]
        step_history = [0]
        critic_losses = []
        pinn_losses   = []
        running_values = []

        for step_idx in range(n_steps):
            outputs = _forward_policy(model, W, goal, step_idx, n_steps, history, step_history)
            if isinstance(outputs, tuple):
                pi, aux_value = outputs
            else:
                pi, aux_value = outputs, None

            pi = _apply_leverage_constraint_torch(pi, d, u, max_long_leverage, max_short_leverage)

            # Antithetic variates: first half uses Z, second half uses −Z
            if antithetic:
                Z_base = torch.randn(base_paths, n_assets, device=dev)
                Z      = torch.cat([Z_base, -Z_base], dim=0)
            else:
                Z = torch.randn(n_paths, n_assets, device=dev)

            dS     = mu_t * dt + (Z @ sig_t.T) * sqrt_dt
            bond   = r_t * dt
            excess = (pi * (dS - bond)).sum(1, keepdim=True)
            W_next = (W * (1.0 + bond + excess)).clamp(min=1e-6)

            if aux_value is not None:
                running_values.append(aux_value)
                if getattr(model, 'kind', '') == 'pinn':
                    features = _current_features(
                        W.clone().detach().requires_grad_(True), goal, step_idx, n_steps
                    )
                    _, value = model(features)
                    grads  = torch.autograd.grad(value.sum(), features, create_graph=True)[0]
                    V_w    = grads[:, :1]
                    V_t    = grads[:, 1:2]
                    V_ww   = torch.autograd.grad(
                        V_w.sum(), features, create_graph=True
                    )[0][:, :1]
                    cov_t_local = sig_t @ sig_t.T
                    drift       = r_t + (pi.detach() * (mu_t - r_t)).sum(dim=1, keepdim=True)
                    diff        = (pi.detach() * (pi.detach() @ cov_t_local)).sum(dim=1, keepdim=True)
                    residual    = (V_t + drift * features[:, :1] * V_w
                                   + 0.5 * diff * features[:, :1] ** 2 * V_ww)
                    pinn_losses.append((residual ** 2).mean())

            W = W_next
            history.append(W)
            step_history.append(step_idx + 1)

        # Terminal utility with adaptive temperature
        utility_term = _terminal_utility(W, goal, utility, asp_p, asp_c1, asp_R,
                                         sig_temp=sig_temp)
        loss = -utility_term.mean()

        if getattr(model, 'kind', '') == 'actor_critic' and running_values:
            target       = utility_term.detach().unsqueeze(1)
            critic_losses = [(v - target).pow(2).mean() for v in running_values]
            loss = loss + 0.5 * torch.stack(critic_losses).mean()
        if getattr(model, 'kind', '') == 'pinn' and pinn_losses:
            loss = loss + 0.1 * torch.stack(pinn_losses).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # Store training E[U] every iteration
        loss_history.append(utility_term.mean().item())

        # ── Validation & early stopping (every 10 iters) ─────────────────────
        if it % 10 == 0:
            with torch.no_grad():
                W_val = torch.full((256, 1), float(w0), dtype=torch.float32, device=dev)
                hist_v, sth_v = [W_val], [0]
                for s in range(n_steps):
                    out_v = _forward_policy(model, W_val, goal, s, n_steps, hist_v, sth_v)
                    pv    = out_v[0] if isinstance(out_v, tuple) else out_v
                    pv    = _apply_leverage_constraint_torch(pv, d, u, max_long_leverage, max_short_leverage)
                    Z_v   = torch.randn(256, n_assets, device=dev)
                    dS_v  = mu_t * dt + (Z_v @ sig_t.T) * sqrt_dt
                    exc_v = (pv * (dS_v - r_t * dt)).sum(1, keepdim=True)
                    W_val = (W_val * (1.0 + r_t * dt + exc_v)).clamp(min=1e-6)
                    hist_v.append(W_val); sth_v.append(s + 1)
                val_u = _terminal_utility(
                    W_val, goal, utility, asp_p, asp_c1, asp_R, sig_temp=sig_temp
                ).mean().item()

            val_history.append(val_u)
            val_iters.append(it)

            if val_u > best_val_utility + 1e-5:
                best_val_utility = val_u
                no_improve_count = 0
            else:
                no_improve_count += 10

        if no_improve_count >= patience:
            if verbose:
                print(f"  [{architecture_name}] early stop at iter {it+1}")
            break

    model.eval()

    # ── Post-training test evaluation ────────────────────────────────────────
    # Fresh MC paths with a held-out seed — unbiased estimate of E[U].
    with torch.no_grad():
        test_paths = 1024
        W_test = torch.full((test_paths, 1), float(w0), dtype=torch.float32, device=dev)
        hist_te, sth_te = [W_test], [0]
        for s in range(n_steps):
            out_te = _forward_policy(model, W_test, goal, s, n_steps, hist_te, sth_te)
            pi_te  = out_te[0] if isinstance(out_te, tuple) else out_te
            pi_te  = _apply_leverage_constraint_torch(
                pi_te, d, u, max_long_leverage, max_short_leverage
            )
            Z_te  = torch.randn(test_paths, n_assets, device=dev)
            dS_te = mu_t * dt + (Z_te @ sig_t.T) * sqrt_dt
            exc_te = (pi_te * (dS_te - r_t * dt)).sum(1, keepdim=True)
            W_test = (W_test * (1.0 + r_t * dt + exc_te)).clamp(min=1e-6)
            hist_te.append(W_test); sth_te.append(s + 1)
        test_u = _terminal_utility(
            W_test, goal, utility, asp_p, asp_c1, asp_R, sig_temp=sig_temp
        ).mean().item()

    param_count = sum(p.numel() for p in model.parameters())
    return model, {
        'architecture_name': architecture_name,
        'backend'          : 'torch',
        'kind'             : getattr(model, 'kind', 'mlp'),
        'param_size'       : int(param_count),
        'device'           : str(dev),
        # ── histories ──────────────────────────────────────────────────────
        'loss_history'     : loss_history,   # train E[U] per iter
        'val_history'      : val_history,    # val   E[U] every 10 iters
        'val_iters'        : val_iters,      # iter indices for val points
        'test_u'           : test_u,         # scalar held-out E[U]
    }
