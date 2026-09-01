# Repository instructions

## Workflow

- Use `uv` and the checked-in `uv.lock`; do not add a second environment or
  dependency manager.
- Keep generated replays, logs, and candidate checkpoints under `dist/`, which
  is git-ignored.
- Do not replace a tracked checkpoint until its checksum, replay, and held-out
  evaluation have passed.
- Keep pull requests as drafts until the source checks and checkpoint validation
  below are complete.

## Checks

Run these before committing source or checkpoint changes:

```bash
uv run python -m lunar_rl.nets
uv lock --check --offline
shasum -a 256 -c checkpoints.sha256
```

There is currently no pytest suite. For training-loop changes, also run one
default PPO iteration plus partial-chunk, pixel, and `torch.compile` smoke runs.

## Robust checkpoint

The production robust recipe trains on the same off-pad distribution used by
the viewer:

```bash
uv run lunar-rl --total-steps 5000000 --num-envs 16 --device mps --seed 0 \
  --start-x 6 --start-tilt 0.5 --save dist/lunar_agent_robust_candidate.pt
```

Evaluate seeds 0–7 and held-out seeds 200–249 greedily before promotion. Confirm
that every episode lands and that both rendered leg endpoints remain between
the helipad flags. After promotion, update `checkpoints.sha256` and every README
command, hash, and metric affected by the new weights.

## Replay

Generate replay HTML without opening a browser unless the user explicitly asks:

```bash
uv run lunar-rl-view --ckpt lunar_agent_robust.pt --episodes 5 --seed 0 --greedy
```

The output is self-contained and normally needs no server. If a local server is
requested, use `python -m http.server`, bind it to `127.0.0.1`, report the exact
URL, and stop the process when asked.
