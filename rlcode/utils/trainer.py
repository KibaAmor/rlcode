import time
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from rlcode.data.experience import Batch, ExperienceSource
from rlcode.policy.policy import Policy


class Trainer:
    def __init__(
        self,
        policy: Policy,
        train_src: ExperienceSource,
        test_src: ExperienceSource,
        writer: Optional[Union[str, SummaryWriter]] = None,
        save_dir: Optional[str] = None,
        eps_collect: float = 0.6,
        eps_collect_decay: float = 0.6,
        eps_collect_min: float = 0.01,
        eps_test: float = 0.01,
    ):
        self._policy = policy
        self._train_src = train_src
        self._test_src = test_src
        self._writer = SummaryWriter(writer) if isinstance(writer, str) else writer
        self._save_dir = save_dir
        self._eps_collect = eps_collect
        self._eps_collect_decay = eps_collect_decay
        self._eps_collect_min = eps_collect_min
        self._eps_test = eps_test
        self._reset()

    def _reset(self) -> None:
        self._epoch = 0

        self._iters = 0
        self._total_iters = 0

        self._collected_steps = 0
        self._sampled_steps = 0

        self._learn_count = 0
        self._total_learn_count = 0

        self._best_rew = -np.inf

    def train(
        self,
        epochs: int,
        iter_per_epoch: int = 1000,
        learn_per_iter: int = 1,
        test_per_epoch: int = 80,
        warmup_collect: int = 0,
        max_reward: Optional[float] = None,
        max_loss: Optional[float] = None,
    ) -> float:
        self._reset()
        self._total_iters = epochs * iter_per_epoch
        self._total_learn_count = self._total_iters * learn_per_iter

        self._policy.train()
        self._warmup(warmup_collect)

        pruned = False
        for _ in range(epochs):
            self._epoch += 1

            with tqdm.tqdm(total=iter_per_epoch, **self._tqdm_config()) as t:
                while t.n < t.total:
                    self._iters += 1
                    t.update()

                    batch, step_per_s = self._collect()
                    loss = self._learn(batch, learn_per_iter)

                    info = {
                        "step/s": f"{step_per_s:.2f}",
                        "loss": f"{loss:.6f}",
                    }
                    t.set_postfix(info)

                    if max_loss is not None and loss > max_loss:
                        pruned = True
                        break

            if pruned:
                break

            rew = -np.inf
            if test_per_epoch > 0:
                self._policy.eval()
                rew = self._test(test_per_epoch)
                self._policy.train()
            self._save(rew)

            if rew > self._best_rew:
                self._best_rew = rew
                if max_reward is not None and rew >= max_reward:
                    break

        if test_per_epoch < 0:
            self._policy.eval()
            rew = self._test(-test_per_epoch)
            self._policy.train()
            self._save(rew)

        return self._best_rew

    def _warmup(self, n: int) -> None:
        if n <= 0:
            return
        self._policy.eps = self._eps_collect
        for _ in range(n):
            batch = self._train_src.collect()
            self._collected_steps += len(batch)

    def _get_collect_eps(self) -> float:
        ratio = self._iters / self._total_iters
        eps = self._eps_collect - (self._eps_collect - self._eps_collect_min) * ratio
        eps = max(eps * self._eps_collect_decay, self._eps_collect_min)
        return eps

    def _collect(self) -> Tuple[Batch, float]:
        eps = self._get_collect_eps()
        self._policy.eps = eps

        beg_t = time.time()
        batch = self._train_src.collect()
        self._collected_steps += len(batch)
        cost_t = max(time.time() - beg_t, 1e-6)
        step_per_s = len(batch) / cost_t

        info = {
            "eps": eps,
            "step/s": step_per_s,
            "dist/act": np.array(batch.acts, copy=False),
        }
        buffer = getattr(self._train_src, "buffer")
        if buffer is not None:
            info["buffer_size"] = len(buffer)
        self._track("collect", info, self._collected_steps)

        return batch, step_per_s

    def _learn(self, batch: Batch, n: int) -> float:
        losses = []
        for _ in range(n):
            info = self._policy.learn(batch, self._train_src)
            info["replay_ratio"] = self._sampled_steps / self._collected_steps
            self._track("learn", info, self._learn_count)

            losses.append(info["loss"])
            self._sampled_steps += info["_batch_size"]
            self._learn_count += 1

        return np.mean(losses)

    def _test(self, n: int) -> float:
        self._policy.eps = self._eps_test

        rews = []
        steps = []
        acts = []
        beg_t = time.time()
        for _ in range(n):
            batch = self._test_src.collect()
            rews.append(sum(batch.rews))
            steps.append(len(batch))
            acts.append(batch.acts)
        cost_t = max(time.time() - beg_t, 1e-6)

        rew_mean = np.mean(rews)
        rew_std = np.std(rews)
        print(f"Epoch #{self._epoch}: {n} test reward={rew_mean:.3f} ± {rew_std:.3f}")

        info = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "dist/rew": np.array(rews, copy=False),
            "step_mean": np.mean(steps),
            "step/s": sum(steps) / cost_t,
            "dist/step": np.array(steps, copy=False),
            "ms/episode": 1000.0 * cost_t / n,
            "dist/act": np.concatenate(acts),
        }
        self._track("test", info, self._epoch)
        return rew_mean

    def _save(self, rew: float) -> None:
        if self._save_dir is None or rew <= self._best_rew:
            return
        policy = deepcopy(self._policy).to(torch.device("cpu"))
        torch.save(policy.state_dict(), f"{self._save_dir}/{rew:.2f}.pth")

    def _track(self, tag: str, info: dict, step: int) -> None:
        if self._writer is None:
            return
        for k, v in info.items():
            if k.startswith("_"):
                continue
            if k.startswith("dist/"):
                self._writer.add_histogram(f"{tag}/{k[5:]}", v, step)
            else:
                self._writer.add_scalar(f"{tag}/{k}", v, step)

    def _tqdm_config(self):
        return dict(
            ascii=True,
            dynamic_ncols=True,
            desc=f"Epoch #{self._epoch}",
        )
