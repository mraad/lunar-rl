# Research baseline TODO

## Context correctness

- [x] Replay every PPO transition with its exact acting window and loss at the
  final position; include history from the previous rollout.
- [x] Evaluate bootstrap without mutating acting history or duplicating tokens.
- [x] Preserve episode masks, initial padding, previous actions/rewards and pixels.
- [x] Gather windows on demand and accumulate gradients with whole-minibatch
  normalization; include every real transition once per epoch.
- [x] Add regression checks using the actual trainer with frozen weights: acting
  and update inputs, logits, values and unit PPO ratios must match across two
  rollouts, resets, partial batches, pixels, and a one-token context.
- [x] Document changed update semantics and mark shipped metrics as historical.

## Validation before retraining

- [x] Run network and context checks, offline lock check, checkpoint checksums,
  and default-iteration, partial-batch, pixel and torch.compile training smokes.
  Completed on 2026-09-08; compile smoke used CPU Inductor. Smoke checkpoints
  are under `dist/context-fix/`; shipped checkpoint checksums still pass.
- [ ] Verify CUDA AMP and compiled accelerator runs on the intended training
  host before treating those configurations as validated.
- [ ] Correct time-limit bootstrapping using terminal observations and pre-reset
  context, with a regression covering SAME_STEP autoreset.
- [ ] Replace reward-based landing classification with explicit terminal outcome,
  both leg endpoints, impact velocity and tilt; record the final post-action state.
- [x] Validate the run-specific terminal/foot evaluator on the shipped robust
  checkpoint: 8/8 and 50/50 pass, held-out mean return +324.25 and minimum
  rendered-foot pad margin 0.571. Results are in
  `dist/robust-context-20260908/baseline/evaluation.json`. Integrating these
  checks into the shared viewer/evaluator remains outstanding.
- [ ] Measure corrected-trainer throughput and peak memory before scheduling
  full training; the old timing tables do not establish current performance.

## Retrain and establish the baseline

- [x] Train a candidate with the robust recipe in AGENTS.md, saved under `dist/`.
  Record source revision, command, seed, device, dependency versions and checksum.
  Completed: `dist/robust-context-20260908/` (seed 0, 16 envs, MPS,
  5,000,000 requested steps). See `status.json` and `train.log`; provenance and
  a source snapshot are saved alongside them. The rollout-aligned total is
  4,999,168 steps. This run still uses the documented time-limit approximation.
  Candidate SHA-256: `2328a5970378b2f47dfae65d1d72c8f62da14287deaeb7af0d4c72a99cb7c68d`.
- [x] Generate a replay under `dist/` without opening a browser; evaluate seeds
  0–7 and 200–249 greedily and confirm every landing and both leg endpoints
  between flags before any checkpoint promotion.
  Completed: terminal-outcome and rendered-foot checks in `evaluation.json`,
  plus an eight-episode `replay.html`. Strict checks passed 5/8 on seeds 0–7 and
  42/50 on seeds 200–249 (mean returns +306.15 and +319.07). The candidate failed
  the all-episodes promotion requirement; shipped checkpoints remain untouched.
- [ ] Diagnose the candidate's failed landings and validate a subsequent run
  before promoting a corrected-context checkpoint as the research baseline.
- [ ] Reserve fresh seeds and disturbance combinations for the research report
  once the existing evaluation seeds have informed controller development.
- [ ] Repeat training across seeds and report landing reliability, worst cases,
  return spread and inference cost; include a simple state-vector baseline.
- [ ] Promote only after validation, then update checkpoints.sha256 and all
  affected README commands, hashes and metrics. Keep PRs draft until checks pass.

## Subsequent control research

- [ ] Define controller-visible observations and evaluator-only simulator truth.
- [ ] Separate vision-only experiments from the current pixels-plus-state mode.
- [x] Implement `research/adaptive-mpc` in `dist/worktrees/adaptive-mpc` with
  online parameter estimation, discrete predictive control, an introductory
  implementation guide, and self-contained replays. Six focused tests pass.
  On seeds 200–249, strict landing checks pass 49/50 for adaptive MPC both
  normally and with a 30% thrust reduction; fixed-model MPC passes 49/50 and
  42/50 respectively. See that worktree's `docs/adaptive-mpc.md` for evidence.
- [ ] Address Adaptive MPC's seed-235 viewport failure, validate recovery
  constraints and deadline handling, then evaluate on a fresh seed set. Current
  penalties/terminal assumptions are approximate and have no safety certificate.
- [ ] Propose `research/hybrid-miqp`: discrete thruster decisions under the same
  observations, actuator limits, scenarios and compute budget as the baseline.
