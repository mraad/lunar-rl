"""Networks for the LunarLander sequence agent.

Three ideas, stacked:
  1. Impala residual CNN  (Espeholt et al. 2018)   -> pixel tokens
  2. GTrXL causal transformer (Parisotto et al. 2020) -> memory over the last K steps
  3. Two-hot symlog critic (Hafner et al. DreamerV3, 2023) -> scale-free value head

The transformer sees a token per timestep built from (state, prev action,
prev reward, optional frame).  Attention is masked so a token never looks
across an episode boundary, which is what makes it safe to train on flat
rollout chunks instead of whole episodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


# --------------------------------------------------------------------------- #
# critic head
# --------------------------------------------------------------------------- #
class TwoHotCritic(nn.Module):
    """Value as classification over symlog-spaced bins.

    Regressing returns with MSE forces a per-environment value scale.  Binning
    in symlog space and training with a two-hot cross-entropy removes that
    tuning knob: the same head handles LunarLander's -400..+300 range and a
    pixel-mode reward-shaped variant without touching hyperparameters.
    """

    def __init__(self, d_model: int, bins: int = 41, lo: float = -20.0, hi: float = 20.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(lo, hi, bins))
        self.fc = nn.Linear(d_model, bins)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)  # start at V=0, no early bootstrap blowup

    def value(self, h: torch.Tensor) -> torch.Tensor:
        p = F.softmax(self.fc(h), dim=-1)
        return symexp((p * self.centers).sum(-1))

    def two_hot(self, y: torch.Tensor) -> torch.Tensor:
        c = self.centers
        y = y.clamp(c[0], c[-1])
        hi = torch.bucketize(y, c).clamp(1, c.numel() - 1)
        lo = hi - 1
        w = ((y - c[lo]) / (c[hi] - c[lo]).clamp_min(1e-8)).unsqueeze(-1)
        out = torch.zeros(*y.shape, c.numel(), device=y.device, dtype=y.dtype)
        out.scatter_(-1, lo.unsqueeze(-1), 1.0 - w)
        out.scatter_add_(-1, hi.unsqueeze(-1), w)
        return out

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(self.fc(h), dim=-1)
        return -(self.two_hot(symlog(target)).detach() * logp).sum(-1)


# --------------------------------------------------------------------------- #
# pixel encoder
# --------------------------------------------------------------------------- #
class ImpalaBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.res = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
            )
            for _ in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv(x), 3, stride=2, padding=1)
        for r in self.res:
            x = x + r(x)
        return x


class PixelEncoder(nn.Module):
    """84x84 grayscale frame -> one d_model token.

    Impala's residual trunk beats the Nature-DQN stack at equal parameter count
    and is the standard choice when the same encoder must survive a long run.
    """

    def __init__(self, d_model: int, in_ch: int = 1, size: int = 84):
        super().__init__()
        self.trunk = nn.Sequential(ImpalaBlock(in_ch, 16), ImpalaBlock(16, 32), ImpalaBlock(32, 32))
        self.head = nn.Sequential(nn.Flatten(), nn.ReLU(), nn.LazyLinear(d_model), nn.LayerNorm(d_model))
        with torch.no_grad():  # materialise the LazyLinear before the optimiser sees params
            self.forward(torch.zeros(1, 1, in_ch, size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        z = self.head(self.trunk(x.flatten(0, 1)))
        return z.view(b, t, -1)


# --------------------------------------------------------------------------- #
# transformer
# --------------------------------------------------------------------------- #
def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (B, H, T, Dh).

    Rotary is *relative*, so a sliding K-step window during acting and a fixed
    K-step chunk during training see identical geometry.  Absolute learned
    positions would not survive that.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class GRUGate(nn.Module):
    """GTrXL gate. Replaces the residual add.

    A vanilla post-LN transformer diverges under PPO's non-stationary targets.
    With bias_init > 0 the update gate starts near zero, so each block begins as
    an identity map and earns its contribution instead of asserting it.
    """

    def __init__(self, d: int, bias_init: float = 2.0):
        super().__init__()
        self.wr, self.ur = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.wz, self.uz = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.wg, self.ug = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.bg = nn.Parameter(torch.full((d,), bias_init))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.wr(y) + self.ur(x))
        z = torch.sigmoid(self.wz(y) + self.uz(x) - self.bg)
        h = torch.tanh(self.wg(y) + self.ug(r * x))
        return torch.lerp(x, h, z)


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.g1, self.g2 = GRUGate(d), GRUGate(d)

    def forward(self, x, mask, rotary, want_attn: bool = False):
        b, t, d = x.shape
        q, k, v = (
            u.view(b, t, self.heads, self.dh).transpose(1, 2)
            for u in self.qkv(self.ln1(x)).chunk(3, dim=-1)
        )
        q, k = rope(q, *rotary), rope(k, *rotary)
        if want_attn:
            # Fused SDPA never materialises the probabilities. The viewer wants
            # them, so this path recomputes attention the slow, explicit way.
            w = (q @ k.transpose(-2, -1)) / self.dh**0.5
            w = w.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
            a = w @ v
        else:
            w = None
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = self.g1(x, self.proj(a.transpose(1, 2).reshape(b, t, d)))
        return self.g2(x, self.ff(self.ln2(x))), w


def episode_mask(starts: torch.Tensor) -> torch.Tensor:
    """(B, T) episode-start flags -> (B, T, T) bool attention mask, True = allowed.

    Causal AND same-episode.  Without the second term a chunk that straddles a
    reset would let the agent condition the new episode on the old one's crash.
    The diagonal is always allowed, so no row is fully masked (no NaN).
    """
    seg = starts.long().cumsum(dim=1)
    same = seg[:, :, None] == seg[:, None, :]
    causal = torch.ones(starts.shape[1], starts.shape[1], dtype=torch.bool, device=starts.device).tril()
    return same & causal


class Agent(nn.Module):
    """Token per timestep -> causal transformer -> actor + critic on every position."""

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 4,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        pixels: bool = False,
    ):
        super().__init__()
        if heads <= 0 or d_model % heads:
            raise ValueError("d_model must be divisible by a positive number of heads")
        head_dim = d_model // heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.register_buffer(
            "rope_freq",
            1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim)),
            persistent=False,
        )
        self.vec = nn.Linear(obs_dim, d_model)
        self.act_emb = nn.Embedding(n_actions + 1, d_model)  # +1 = "no previous action"
        self.rew = nn.Linear(1, d_model)
        self.pix = PixelEncoder(d_model) if pixels else None
        self.ln_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.actor = nn.Linear(d_model, n_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)  # near-uniform policy at init
        nn.init.zeros_(self.actor.bias)
        self.critic = TwoHotCritic(d_model)

    def forward(self, obs, prev_act, prev_rew, starts, frames=None, want_attn: bool = False):
        """obs (B,T,obs_dim) | prev_act (B,T) long | prev_rew (B,T) | starts (B,T) bool.

        Returns (logits, h), or (logits, h, [attn per layer]) when want_attn.
        """
        tok = self.vec(obs) + self.act_emb(prev_act) + self.rew(symlog(prev_rew).unsqueeze(-1))
        if self.pix is not None:
            tok = tok + self.pix(frames)
        h = self.ln_in(tok)
        mask = episode_mask(starts)
        pos = torch.arange(obs.shape[1], device=obs.device)
        ang = pos.to(h.dtype)[:, None] * self.rope_freq.to(h.dtype)
        rotary = ang.cos(), ang.sin()
        attns = []
        for blk in self.blocks:
            h, w = blk(h, mask, rotary, want_attn)
            if want_attn:
                attns.append(w)
        h = self.ln_out(h)
        if want_attn:
            return self.actor(h), h, attns
        return self.actor(h), h


# --------------------------------------------------------------------------- #
def _demo() -> None:
    torch.manual_seed(0)
    net = Agent(pixels=True)

    b, t = 2, 12
    obs = torch.randn(b, t, 8)
    pa = torch.randint(0, 5, (b, t))
    pr = torch.randn(b, t)
    frames = torch.rand(b, t, 1, 84, 84)
    starts = torch.zeros(b, t, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 6] = True  # a reset mid-chunk

    logits, h = net(obs, pa, pr, starts, frames)
    assert logits.shape == (b, t, 4) and h.shape == (b, t, 128)

    # the explicit-attention path must agree with the fused kernel
    logits_w, _, attns = net(obs, pa, pr, starts, frames, want_attn=True)
    assert torch.allclose(logits, logits_w, atol=1e-4), "want_attn path diverged from SDPA"
    assert len(attns) == 3 and attns[0].shape == (b, 4, t, t)
    assert torch.allclose(attns[-1].sum(-1), torch.ones(b, 4, t), atol=1e-5)
    assert attns[-1][:, :, 8, :6].abs().max() < 1e-9, "attention crossed the reset"

    # episode isolation: perturbing episode 1 must not move episode 2's outputs
    obs2 = obs.clone()
    obs2[:, :6] += 10.0
    frames2 = frames.clone()
    frames2[:, :6] = 0.0
    logits2, _ = net(obs2, pa, pr, starts, frames2)
    assert torch.allclose(logits[:, 6:], logits2[:, 6:], atol=1e-5), "attention leaked across reset"
    assert not torch.allclose(logits[:, :6], logits2[:, :6]), "perturbation had no effect at all"

    # two-hot is a distribution and its expectation reconstructs the target
    y = torch.tensor([-3.7, 0.0, 12.4])
    p = net.critic.two_hot(y)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.allclose((p * net.critic.centers).sum(-1), y, atol=1e-5)
    assert torch.allclose(symexp(symlog(y)), y, atol=1e-5)

    # value head starts at exactly 0 (zero-init logits -> uniform -> mean of centers = 0)
    assert net.critic.value(h).abs().max() < 1e-5

    print("nets self-check OK")


if __name__ == "__main__":
    _demo()
