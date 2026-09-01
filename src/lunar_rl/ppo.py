"""PPO over a causal transformer, with chunked BPTT and R2D2-style burn-in.

Rollout shape: (T, N) transitions from N parallel envs.  For the update the
rollout is cut into contiguous chunks of `chunk` steps; each chunk is prefixed
by `burn_in` earlier steps whose only job is to fill attention context.  The
loss is masked off on the burn-in prefix, so gradients only touch positions
whose context matches what the actor actually saw.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import pathlib
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical

from lunar_rl.nets import Agent

NO_ACTION = 4  # embedding index for "episode just started, no previous action"


# --------------------------------------------------------------------------- #
class PixelObs(gym.ObservationWrapper):
    """Adds an 84x84 grayscale render next to the state vector.

    A single frame hides velocity and angular rate, which is exactly the partial
    observability the transformer context exists to fix.  The state vector stays
    in the observation so vector-only and pixel runs share one code path.
    """

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Dict(
            {
                "state": env.observation_space,
                "pixels": spaces.Box(0.0, 1.0, (1, size, size), dtype=np.float32),
            }
        )
        self._yi = self._xi = None

    def observation(self, obs):
        frame = self.env.render()
        if self._yi is None:  # ponytail: nearest-neighbour resize, avoids an opencv dep
            self._yi = np.linspace(0, frame.shape[0] - 1, self.size).astype(np.int64)
            self._xi = np.linspace(0, frame.shape[1] - 1, self.size).astype(np.int64)
        gray = frame.astype(np.float32).mean(-1) / 255.0
        return {"state": obs, "pixels": gray[self._yi][:, self._xi][None]}



class StartPose(gym.Wrapper):
    """Spawn the lander off the pad and tilted.

    Vanilla LunarLander always spawns dead centre above the helipad at angle 0;
    the only per-seed randomness is a small force impulse, so **no choice of seed
    varies the start pose**.  This rigidly transforms the lander and both legs
    after reset — rigid so the revolute joints stay satisfied — and then
    re-derives the observation exactly the way the env does, since
    `LunarLander.reset` itself ends with an idle step.

    Draws from the env's own RNG, so the pose is a deterministic function of the
    seed: seed N now genuinely means a different approach, not just different
    terrain.
    """

    def __init__(self, env: gym.Env, x_range: float = 0.0, tilt_range: float = 0.0):
        super().__init__(env)
        self.x_range, self.tilt_range = x_range, tilt_range
        self.start = (0.0, 0.0)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        u = self.env.unwrapped
        # Magnitude floor at 40% of range: a plain uniform draw would sometimes
        # land back on the pad (half-width 2.0) at a tilt too small to see.
        def draw(rng: float) -> float:
            if not rng:
                return 0.0
            sign = 1.0 if u.np_random.random() < 0.5 else -1.0
            return sign * float(u.np_random.uniform(0.4 * rng, rng))

        dx, tilt = draw(self.x_range), draw(self.tilt_range)
        self.start = (dx, tilt)
        if dx or tilt:
            cx, cy = u.lander.position.x, u.lander.position.y
            c, s = math.cos(tilt), math.sin(tilt)
            for b in [u.lander, *u.legs]:
                px, py = b.position.x - cx, b.position.y - cy
                b.position = (cx + dx + c * px - s * py, cy + s * px + c * py)
                b.angle += tilt
            idle = np.array([0.0, 0.0], dtype=np.float32) if u.continuous else 0
            obs = u.step(idle)[0]
        info["start"] = self.start
        return obs, info


def make_envs(n: int, pixels: bool, seed: int, start_x: float = 0.0,
              start_tilt: float = 0.0) -> gym.vector.VectorEnv:
    def thunk(idx: int):
        def _f():
            env = gym.make("LunarLander-v3", render_mode="rgb_array" if pixels else None)
            if start_x or start_tilt:  # innermost, so it sees the raw env
                env = StartPose(env, start_x, start_tilt)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if pixels:
                env = PixelObs(env)
            env.reset(seed=seed + idx)
            return env

        return _f

    cls = gym.vector.AsyncVectorEnv if pixels else gym.vector.SyncVectorEnv
    # SAME_STEP autoreset keeps `done[t]` aligned with the transition at t, which
    # is what the GAE recursion and the episode mask below both assume.
    return cls([thunk(i) for i in range(n)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


# --------------------------------------------------------------------------- #
@dataclass
class Config:
    total_steps: int = 1_000_000
    num_envs: int = 16
    rollout: int = 128
    chunk: int = 24
    burn_in: int = 8
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    pixels: bool = False
    seed: int = 1
    device: str = "auto"
    compile: bool = False   # torch.compile the agent (CUDA: big win on this model size)
    amp: bool = False       # bf16 autocast (CUDA only)
    save: str = "lunar_agent.pt"
    start_x: float = 0.0     # spawn offset magnitude, world units; pad half-width is 2.0
    start_tilt: float = 0.0  # spawn tilt magnitude, radians (0 = vanilla)

    @property
    def ctx(self) -> int:
        return self.burn_in + self.chunk

    def __post_init__(self) -> None:
        if self.chunk <= 0 or self.rollout <= 0 or self.burn_in < 0:
            raise ValueError("rollout and chunk must be positive; burn_in cannot be negative")
        if self.num_envs <= 0 or self.total_steps < self.rollout * self.num_envs:
            raise ValueError("total_steps must cover at least one rollout across all environments")
        if self.minibatches <= 0 or self.epochs <= 0:
            raise ValueError("minibatches and epochs must be positive")


def pin_bundled_cudnn(device: torch.device) -> None:
    """Make the wheel's cuDNN win the soname race against a system CUDA install.

    Some CUDA hosts ship a cuDNN built against CUDA 12 on `LD_LIBRARY_PATH`.  It
    carries the same `libcudnn_*.so.9` soname as the cu13 wheel torch depends on,
    so the loader hands the conv engines the system copy, which then dlopens
    `libcublasLt.so.12` -- absent on a CUDA 13 box.  The process aborts in native
    code ("Cannot load symbol cublasLtGetVersion") the first time a convolution
    runs, so it hits `--pixels` only: the vector path has no CNN.

    dlopen-ing the bundled copies by absolute path first means the later
    resolve-by-soname finds them already loaded.  Silent no-op when cuDNN is not
    bundled, which is the normal case off Linux.
    """
    if device.type != "cuda":
        return
    try:
        import nvidia  # namespace package, so __path__ rather than __file__
        lib = pathlib.Path(list(nvidia.__path__)[0]) / "cudnn" / "lib"
    except Exception:
        return
    for so in sorted(lib.glob("libcudnn*.so.9")):
        try:
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
class History:
    """Rolling per-env context window used only while acting."""

    def __init__(self, n: int, ctx: int, obs_dim: int, pixels: bool, device: torch.device):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.obs = z(n, ctx, obs_dim)
        self.act = z(n, ctx, dt=torch.long).fill_(NO_ACTION)
        self.rew = z(n, ctx)
        self.start = z(n, ctx, dt=torch.bool)
        self.pix = z(n, ctx, 1, 84, 84) if pixels else None

    def push(self, obs, prev_act, prev_rew, start, pix=None):
        for name in ("obs", "act", "rew", "start", "pix"):
            buf = getattr(self, name)
            if buf is not None:
                setattr(self, name, torch.roll(buf, -1, dims=1))
        self.obs[:, -1], self.act[:, -1], self.rew[:, -1], self.start[:, -1] = obs, prev_act, prev_rew, start
        if self.pix is not None:
            self.pix[:, -1] = pix


def split_obs(raw, device, pixels):
    if pixels:
        state = torch.as_tensor(raw["state"], dtype=torch.float32, device=device)
        frame = torch.as_tensor(raw["pixels"], dtype=torch.float32, device=device)
        return state, frame
    return torch.as_tensor(raw, dtype=torch.float32, device=device), None


# --------------------------------------------------------------------------- #
def train(cfg: Config) -> Agent:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    if device.type == "cuda":  # Ampere+ tensor cores for the fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = cfg.amp and device.type == "cuda"
    if cfg.amp and not amp:
        print("--amp ignored: bf16 autocast is CUDA-only here", flush=True)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    if cfg.pixels:  # the CNN is the only user of cuDNN
        pin_bundled_cudnn(device)
    envs = make_envs(cfg.num_envs, cfg.pixels, cfg.seed, cfg.start_x, cfg.start_tilt)
    obs_dim = 8
    N, T, C = cfg.num_envs, cfg.rollout, cfg.ctx

    agent = Agent(obs_dim, 4, cfg.d_model, cfg.layers, cfg.heads, cfg.pixels).to(device)
    # compile the callable, keep `agent` for state_dict and the critic helpers
    fwd = torch.compile(agent) if cfg.compile else agent
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.lr, eps=1e-5)

    buf = {
        "obs": torch.zeros(T, N, obs_dim, device=device),
        "act": torch.zeros(T, N, dtype=torch.long, device=device),
        "prev_act": torch.zeros(T, N, dtype=torch.long, device=device),
        "prev_rew": torch.zeros(T, N, device=device),
        "start": torch.zeros(T, N, dtype=torch.bool, device=device),
        "logp": torch.zeros(T, N, device=device),
        "val": torch.zeros(T, N, device=device),
        "rew": torch.zeros(T, N, device=device),
        "done": torch.zeros(T, N, device=device),
    }
    if cfg.pixels:
        buf["pix"] = torch.zeros(T, N, 1, 84, 84, device=device)

    raw, _ = envs.reset(seed=cfg.seed)
    obs, frame = split_obs(raw, device, cfg.pixels)
    prev_act = torch.full((N,), NO_ACTION, dtype=torch.long, device=device)
    prev_rew = torch.zeros(N, device=device)
    start = torch.ones(N, dtype=torch.bool, device=device)
    hist = History(N, C, obs_dim, cfg.pixels, device)

    # training-window index map: each chunk plus its burn-in prefix
    heads = np.arange(0, T, cfg.chunk)
    offsets = np.arange(-cfg.burn_in, cfg.chunk)
    indices = heads[:, None] + offsets[None, :]
    # Clipped slots only pad the first burn-in and final partial chunk to C;
    # the mask keeps every synthetic duplicate out of all losses.
    valid = (indices >= 0) & (indices < T) & (offsets[None, :] >= 0)
    win = torch.as_tensor(np.clip(indices, 0, T - 1), device=device)
    loss_mask = torch.as_tensor(valid, dtype=torch.float32, device=device)
    loss_mask = loss_mask[:, None].expand(-1, N, -1).flatten(0, 1)

    iters = cfg.total_steps // (T * N)
    returns, step, t0 = [], 0, time.time()

    for it in range(iters):
        for g in opt.param_groups:  # linear anneal, standard PPO
            g["lr"] = cfg.lr * (1.0 - it / iters)

        agent.eval()
        for t in range(T):
            hist.push(obs, prev_act, prev_rew, start, frame)
            with torch.inference_mode(), autocast():
                logits, h = fwd(hist.obs, hist.act, hist.rew, hist.start, hist.pix)
                dist = Categorical(logits=logits[:, -1].float())
                action = dist.sample()
                value = agent.critic.value(h[:, -1].float())

            buf["obs"][t], buf["act"][t], buf["prev_act"][t] = obs, action, prev_act
            buf["prev_rew"][t], buf["start"][t] = prev_rew, start
            buf["logp"][t], buf["val"][t] = dist.log_prob(action), value
            if cfg.pixels:
                buf["pix"][t] = frame

            raw, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            obs, frame = split_obs(raw, device, cfg.pixels)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device)
            buf["rew"][t] = reward_t
            buf["done"][t] = torch.as_tensor(done, dtype=torch.float32, device=device)

            start = torch.as_tensor(done, dtype=torch.bool, device=device)
            prev_act = torch.where(start, torch.full_like(action, NO_ACTION), action)
            prev_rew = torch.where(start, torch.zeros_like(reward_t), reward_t)
            step += N

            # SAME_STEP autoreset parks the finished episode's stats under final_info
            ep = info.get("final_info", info).get("episode")
            if ep is not None:
                returns.extend(np.asarray(ep["r"])[np.asarray(ep["_r"])].tolist())

        with torch.inference_mode():  # bootstrap from the state after the last stored step
            hist.push(obs, prev_act, prev_rew, start, frame)
            _, h = fwd(hist.obs, hist.act, hist.rew, hist.start, hist.pix)
            last_val = agent.critic.value(h[:, -1].float())

        # GAE. ponytail: truncation is treated as termination (LunarLander only
        # truncates at 1000 steps, after the policy has long since landed or crashed).
        adv = torch.zeros_like(buf["rew"])
        run = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["done"][t]
            nxt = last_val if t == T - 1 else buf["val"][t + 1]
            delta = buf["rew"][t] + cfg.gamma * nxt * nonterm - buf["val"][t]
            run = delta + cfg.gamma * cfg.gae_lambda * nonterm * run
            adv[t] = run
        ret = adv + buf["val"]

        # (T, N, ...) -> (nchunk * N, C, ...) training windows
        def windows(x):
            w = x[win]                                   # (nchunk, C, N, ...)
            return w.permute(0, 2, 1, *range(3, w.dim())).flatten(0, 1)

        wobs, wpa, wpr, wst = (windows(buf[k]) for k in ("obs", "prev_act", "prev_rew", "start"))
        wact, wlogp, wadv, wret = (windows(x) for x in (buf["act"], buf["logp"], adv, ret))
        wpix = windows(buf["pix"]) if cfg.pixels else None
        B = wobs.shape[0]
        mb = max(1, B // cfg.minibatches)

        agent.train()
        for _ in range(cfg.epochs):
            for i in torch.randperm(B, device=device).split(mb):
                with autocast():
                    logits, h = fwd(wobs[i], wpa[i], wpr[i], wst[i], None if wpix is None else wpix[i])
                logits, h = logits.float(), h.float()
                dist = Categorical(logits=logits)
                m = loss_mask[i]
                n = m.sum().clamp_min(1.0)

                a = wadv[i]
                a = a - (a * m).sum() / n
                a = a / ((a.square() * m).sum().div(n).sqrt() + 1e-8)
                ratio = (dist.log_prob(wact[i]) - wlogp[i]).exp()
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))

                pg_loss = (pg * m).sum() / n
                v_loss = (agent.critic.loss(h, wret[i]) * m).sum() / n
                ent = (dist.entropy() * m).sum() / n
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()

        if returns:
            recent = returns[-50:]
            print(
                f"iter {it + 1}/{iters}  step {step}  return {np.mean(recent):7.1f}"
                f"  eps {len(returns)}  sps {step / (time.time() - t0):.0f}",
                flush=True,
            )
    envs.close()
    save = pathlib.Path(cfg.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg.__dict__, "model": agent.state_dict()}, save)
    print(f"saved {save}", flush=True)
    return agent


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Train the LunarLander sequence agent.")
    for field, default in Config().__dict__.items():
        flag = f"--{field.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true", default=default)
        else:
            p.add_argument(flag, type=type(default), default=default)
    cfg = Config(**vars(p.parse_args()))
    print(cfg, flush=True)
    train(cfg)


if __name__ == "__main__":
    main()
