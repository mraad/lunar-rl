"""Run with: uv run python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lunar_rl.nets import Agent
from lunar_rl.ppo import Config, History, train


class ContextTest(unittest.TestCase):
    def test_bootstrap_does_not_advance_history(self):
        for ctx in (1, 5, 32):
            hist = History(2, ctx, 8, True, torch.device("cpu"))
            for t in range(ctx + 2):
                values = (torch.full((2, 8), float(t)), torch.full((2,), t % 4),
                          torch.full((2,), float(t)), torch.tensor([t == 0, t == 2]),
                          torch.full((2, 1, 84, 84), float(t)))
                before = tuple(x.clone() for x in
                               (hist.obs, hist.act, hist.rew, hist.start, hist.pix))
                peek = hist.window(*values)
                for old, current in zip(before, (hist.obs, hist.act, hist.rew, hist.start, hist.pix)):
                    torch.testing.assert_close(old, current, rtol=0, atol=0)
                hist.push(*values)
                for expected, actual in zip(peek, (hist.obs, hist.act, hist.rew, hist.start, hist.pix)):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                if ctx > 1 and t:
                    self.assertEqual(hist.obs[0, -2, 0].item(), t - 1)

    def test_training_replays_each_acting_context(self):
        # Exercise the real training loop, including updates, two rollouts,
        # reset boundaries, short rollouts, partial microbatches and pixels.
        torch.set_num_threads(1)
        for pixels, chunk, burn_in in ((False, 24, 8), (False, 1, 0), (True, 3, 2)):
            with self.subTest(pixels=pixels, chunk=chunk, burn_in=burn_in):
                cfg = Config(total_steps=28, num_envs=2, rollout=7, chunk=chunk,
                             burn_in=burn_in, pixels=pixels, d_model=16, layers=2,
                             heads=2, epochs=2, minibatches=3, lr=0, device="cpu")
                expected, visits = {}, {}
                case = self

                class Envs:
                    t = 0

                    def observation(self):
                        obs = np.full((2, 8), self.t, dtype=np.float32)
                        obs[:, 0] = self.t * 2 + np.arange(2)
                        if pixels:
                            return {"state": obs, "pixels": np.full(
                                (2, 1, 84, 84), self.t / 20, dtype=np.float32)}
                        return obs

                    def reset(self, seed):
                        return self.observation(), {}

                    def step(self, actions):
                        self.t += 1
                        # Resets inside a rollout, exactly at its boundary,
                        # and after it. One truncation also resets history.
                        term = np.array([self.t in (3, 7), self.t == 8])
                        trunc = np.array([False, self.t == 5])
                        return self.observation(), np.array([0.25, -0.5]), term, trunc, {}

                    def close(self):
                        pass

                class CheckedAgent(Agent):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        # A nonzero critic makes value parity meaningful even
                        # with lr=0 (the production head starts at zero).
                        torch.nn.init.normal_(self.critic.fc.weight, std=0.02)

                    def forward(self, *args, **kwargs):
                        logits, h = super().forward(*args, **kwargs)
                        for row in range(args[0].shape[0]):
                            key = int(args[0][row, -1, 0])
                            context = tuple(x[row].detach().clone() if x is not None else None
                                            for x in args)
                            current = (context, logits[row, -1].detach(),
                                       self.critic.value(h[row, -1]).detach())
                            if not self.training and key not in expected:
                                expected[key] = current
                            else:
                                previous = expected[key]
                                for old, new in zip(previous[0], context):
                                    if old is None:
                                        case.assertIsNone(new)
                                    else:
                                        torch.testing.assert_close(old, new, rtol=0, atol=0)
                                # lr=0 freezes weights: same contexts must yield
                                # the same actor, critic, and PPO ratio of one.
                                torch.testing.assert_close(previous[1], current[1], rtol=1e-5, atol=1e-6)
                                torch.testing.assert_close(previous[2], current[2], rtol=1e-5, atol=1e-6)
                                ratio = (current[1].log_softmax(-1) - previous[1].log_softmax(-1)).exp()
                                torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-5, atol=1e-6)
                            if self.training:
                                visits[key] = visits.get(key, 0) + 1
                        return logits, h

                Path("dist").mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir="dist") as tmp:
                    cfg.save = str(Path(tmp) / "context-test.pt")
                    with patch("lunar_rl.ppo.make_envs", return_value=Envs()), \
                         patch("lunar_rl.ppo.Agent", CheckedAgent):
                        train(cfg)
                self.assertEqual(visits, {i: cfg.epochs for i in range(cfg.total_steps)})

    def test_gradient_accumulation_preserves_update(self):
        # Same context length, rollout and optimizer minibatches; only the
        # forward/backward batch cap changes, including an uneven final slice.
        torch.set_num_threads(1)
        Path("dist").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="dist") as tmp:
            agents = [train(Config(
                total_steps=14, num_envs=2, rollout=7, chunk=chunk, burn_in=6-chunk,
                d_model=16, layers=2, heads=2, epochs=2, minibatches=3, device="cpu",
                save=str(Path(tmp) / f"batch-{chunk}.pt"),
            )) for chunk in (6, 2)]
        for name, value in agents[0].state_dict().items():
            torch.testing.assert_close(value, agents[1].state_dict()[name], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
