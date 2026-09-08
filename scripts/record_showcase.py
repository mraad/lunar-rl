"""Record one real policy landing as MP4 plus a README-friendly GIF preview."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import gymnasium as gym
import numpy as np
import torch

from lunar_rl.nets import Agent
from lunar_rl.ppo import (NO_ACTION, Config, History, PixelObs, StartPose,
                          pick_device, pin_bundled_cudnn)
from lunar_rl.viewer import to_batch


class Encoder:
    def __init__(self, output: Path, frame: np.ndarray):
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.process = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", "50", "-i", "-", "-vf", "scale=720:-2:flags=lanczos",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read().decode()
        if self.process.wait():
            raise RuntimeError(error)
        preview = self.output.with_suffix(".gif")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(self.output),
            "-vf", "fps=10,scale=480:-2:flags=lanczos,split[a][b];"
                   "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer",
            "-loop", "0", str(preview),
        ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="lunar_agent_robust.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("docs/showcase.mp4"))
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required to encode the showcase")

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**blob["cfg"])
    device = pick_device(args.device)
    if cfg.pixels:
        pin_bundled_cudnn(device)
    agent = Agent(8, 4, cfg.d_model, cfg.layers, cfg.heads, cfg.pixels).to(device).eval()
    agent.load_state_dict(blob["model"])
    env = StartPose(gym.make("LunarLander-v3", render_mode="rgb_array"), 6, .5)
    if cfg.pixels:
        env = PixelObs(env)
    raw, _ = env.reset(seed=args.seed)
    obs, pixels = to_batch(raw, device, cfg.pixels)
    hist = History(1, cfg.ctx, 8, cfg.pixels, device)
    prev_act = torch.tensor([NO_ACTION], device=device)
    prev_rew = torch.zeros(1, device=device)
    start = torch.ones(1, dtype=torch.bool, device=device)
    first = env.render()
    encoder = Encoder(args.out, first)
    total = 0.0
    steps = 0
    try:
        encoder.write(first)
        done = False
        while not done:
            hist.push(obs, prev_act, prev_rew, start, pixels)
            with torch.inference_mode():
                logits, _ = agent(hist.obs, hist.act, hist.rew, hist.start, hist.pix)
                action = logits[0, -1].argmax()
            raw, reward, terminated, truncated, _ = env.step(int(action))
            frame = env.render()
            encoder.write(frame)
            done = terminated or truncated
            total += reward
            steps += 1
            obs, pixels = to_batch(raw, device, cfg.pixels)
            prev_act = torch.tensor([int(action)], device=device)
            prev_rew = torch.tensor([reward], dtype=torch.float32, device=device)
            start = torch.zeros(1, dtype=torch.bool, device=device)
        for _ in range(25):
            encoder.write(frame)
        if not terminated or total <= 0:
            raise RuntimeError("seed did not finish as a successful landing")
    finally:
        env.close()
        encoder.close()
    print(f"Recorded seed {args.seed}: return {total:.1f}, {steps} steps")
    print(f"Wrote {args.out} and {args.out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
