"""Roll out a trained agent and emit a self-contained replay page.

The page is a single HTML file with the trajectories inlined, so it opens over
file://, survives being emailed, and needs no server and no build step.  The
agent runs here, in Python; the page only replays what it recorded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import webbrowser

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.box2d import lunar_lander as LL
from torch.distributions import Categorical

from lunar_rl.nets import Agent
from lunar_rl.ppo import NO_ACTION, Config, History, PixelObs, StartPose, pick_device

TEMPLATE = pathlib.Path(__file__).with_name("replay.html")
ACTIONS = ["idle", "left engine", "main engine", "right engine"]


def _r(x, n: int = 3):
    return round(float(x), n)


def ground_line(env: gym.Env) -> list[list[float]]:
    """Terrain top edge, recovered from the sky polygons the env builds on reset."""
    polys = env.unwrapped.sky_polys
    pts = [[_r(p[0][0], 2), _r(p[0][1], 2)] for p in polys]
    pts.append([_r(polys[-1][1][0], 2), _r(polys[-1][1][1], 2)])
    return pts


def to_batch(raw, device, pixels):
    """Single-env observation -> the (1, ...) batch shape History expects."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)[None]
    return (t(raw["state"]), t(raw["pixels"])) if pixels else (t(raw), None)


def record(ckpt: str, episodes: int, seed: int, greedy: bool, device_name: str,
           start_x: float = 0.0, start_tilt: float = 0.0) -> dict:
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**blob["cfg"])
    device = pick_device(device_name)
    agent = Agent(8, 4, cfg.d_model, cfg.layers, cfg.heads, cfg.pixels).to(device).eval()
    agent.load_state_dict(blob["model"])

    env = gym.make("LunarLander-v3", render_mode="rgb_array" if cfg.pixels else None)
    if start_x or start_tilt:
        env = StartPose(env, start_x, start_tilt)
    if cfg.pixels:
        env = PixelObs(env)

    out = []
    for ep in range(episodes):
        raw, info0 = env.reset(seed=seed + ep)
        obs, frame = to_batch(raw, device, cfg.pixels)
        hist = History(1, cfg.ctx, 8, cfg.pixels, device)
        prev_act = torch.tensor([NO_ACTION], device=device)
        prev_rew = torch.zeros(1, device=device)
        start = torch.ones(1, dtype=torch.bool, device=device)

        states, acts, probs, rews, vals, attn = [], [], [], [], [], []
        total, done = 0.0, False
        while not done:
            hist.push(obs, prev_act, prev_rew, start, frame)
            with torch.no_grad():
                logits, h, ws = agent(
                    hist.obs, hist.act, hist.rew, hist.start, hist.pix, want_attn=True
                )
                p = logits[0, -1].softmax(-1)
                a = p.argmax() if greedy else Categorical(probs=p).sample()
                v = agent.critic.value(h[:, -1])[0]

            states.append([_r(x, 4) for x in hist.obs[0, -1].tolist()])
            acts.append(int(a))
            probs.append([_r(x) for x in p.tolist()])
            vals.append(_r(v))
            # last layer, head-mean, attention paid by the current step to its window
            attn.append([_r(x, 4) for x in ws[-1][0, :, -1, :].mean(0).tolist()])

            raw, reward, term, trunc, _ = env.step(int(a))
            done = term or trunc
            total += reward
            rews.append(_r(reward, 2))
            obs, frame = to_batch(raw, device, cfg.pixels)
            prev_act = torch.tensor([int(a)], device=device)
            prev_rew = torch.tensor([reward], dtype=torch.float32, device=device)
            start = torch.zeros(1, dtype=torch.bool, device=device)

        out.append(
            {
                "seed": seed + ep,
                "start": [_r(v) for v in info0.get("start", (0.0, 0.0))],
                "terrain": ground_line(env),
                "helipad": [env.unwrapped.helipad_x1, env.unwrapped.helipad_x2,
                            _r(env.unwrapped.helipad_y, 3)],
                "return": _r(total, 1),
                "landed": bool(term and total > 0),
                "state": states, "act": acts, "prob": probs,
                "rew": rews, "val": vals, "attn": attn,
            }
        )
        sx, stilt = out[-1]["start"]
        print(f"episode {ep}  start x{sx:+6.2f} tilt{np.degrees(stilt):+6.1f}deg"
              f"  return {total:7.1f}  steps {len(acts)}", flush=True)
    env.close()

    return {
        "meta": {
            "ckpt": ckpt,
            "ctx": cfg.ctx,
            "burn_in": cfg.burn_in,
            "pixels": cfg.pixels,
            "policy": "greedy" if greedy else "sampled",
            "start_x": start_x,
            "start_tilt": start_tilt,
            "world": [LL.VIEWPORT_W / LL.SCALE, LL.VIEWPORT_H / LL.SCALE],
            "leg_down": LL.LEG_DOWN / LL.SCALE,
            "lander_poly": [[x / LL.SCALE, y / LL.SCALE] for x, y in LL.LANDER_POLY],
            "leg": [LL.LEG_AWAY / LL.SCALE, LL.LEG_DOWN / LL.SCALE,
                    LL.LEG_W / LL.SCALE, LL.LEG_H / LL.SCALE],
            "actions": ACTIONS,
        },
        "episodes": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Record episodes and build the replay page.")
    p.add_argument("--ckpt", default="lunar_agent.pt")
    p.add_argument("--out", default="lunar_replay.html")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    p.add_argument("--device", default="auto")
    p.add_argument("--start-x", type=float, default=6.0,
                   help="spawn offset from the pad, world units (pad spans +-2)")
    p.add_argument("--start-tilt", type=float, default=0.5,
                   help="spawn tilt, radians (0.5 = +-29 deg)")
    p.add_argument("--open", action="store_true", help="open the page when done")
    a = p.parse_args()

    data = record(a.ckpt, a.episodes, a.seed, a.greedy, a.device, a.start_x, a.start_tilt)
    page = TEMPLATE.read_text().replace(
        "__REPLAY_DATA__", json.dumps(data, separators=(",", ":"))
    )
    out = pathlib.Path(a.out)
    out.write_text(page)
    mean = np.mean([e["return"] for e in data["episodes"]])
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  mean return {mean:.1f}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
