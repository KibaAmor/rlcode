# -*- coding: utf-8 -*-
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from rlcode.data.collector import Collector
from rlcode.data.types import Batch, Data
from rlcode.policy.policy import Policy


class Trainer(ABC):
    def __init__(
        self,
        policy: Policy,
        train_collector: Collector,
        test_collector: Collector,
        log_dir: str,
        writer: Optional[SummaryWriter],
        eps_collect: float,
        eps_collect_decay: float,
        eps_collect_min: float,
        eps_test: float,
    ):
        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir, flush_secs=60) if writer is None else writer
        self.eps_collect = eps_collect
        self.eps_collect_decay = eps_collect_decay
        self.eps_collect_min = eps_collect_min
        self.eps_test = eps_test
        self.reset()

    def reset(self) -> None:
        self.iter = 0
        self.total_iter = 0
        self.learn_count = 0
        self.total_learn_count = 0
        self.epoch = 0
        self.best_rew = -np.inf
        self.collected_step = 0
        self.sampled_step = 0

    def train(
        self,
        epochs: int,
        iter_per_epoch: int = 1000,
        learn_per_iter: int = 1,
        test_per_epoch: int = 80,
        max_reward: Optional[float] = None,
        max_loss: Optional[float] = None,
        **kwargs,
    ) -> float:
        self.reset()
        self.total_iter = epochs * iter_per_epoch
        self.total_learn_count = self.total_iter * learn_per_iter

        pruned = False
        for _ in range(epochs):
            self.epoch += 1

            with tqdm.tqdm(total=iter_per_epoch, **self.tqdm_cfg()) as t:
                while t.n < t.total:
                    self.iter += 1
                    t.update()

                    dat, step_per_s = self.collect(**kwargs)
                    loss = self.learn(dat, learn_per_iter)
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
                rew = self.test(test_per_epoch)
            self.save(rew)

            if rew > self.best_rew:
                self.best_rew = rew
                if max_reward is not None and rew >= max_reward:
                    break

        if test_per_epoch < 0:
            rew = self.test(-test_per_epoch)
            self.save(rew)

        if self.writer is not None:
            self.writer.close()

        return self.best_rew

    @abstractmethod
    def do_collect(self, **kwargs) -> Tuple[Data, dict]:
        raise NotImplementedError

    def collect(self, **kwargs) -> Tuple[Data, float]:
        def get_collect_eps() -> float:
            ratio = self.iter / self.total_iter
            eps = self.eps_collect - (self.eps_collect - self.eps_collect_min) * ratio
            eps = max(eps * self.eps_collect_decay, self.eps_collect_min)
            return eps

        eps = get_collect_eps()
        self.policy.eps = eps
        self.policy.eval()

        beg_t = time.time()
        dat, info = self.do_collect(**kwargs)
        cost_t = max(time.time() - beg_t, 1e-6)
        step_per_s = len(dat) / cost_t

        info["eps"] = eps
        info["step/s"] = step_per_s

        self.collected_step += len(dat)
        self.track("collect", info, self.collected_step)

        return dat, step_per_s

    @abstractmethod
    def do_learn(self, dat: Data, collector: Collector) -> Tuple[int, float, dict]:
        raise NotImplementedError

    def learn(self, dat: Data, num: int) -> float:
        self.policy.train()

        losses = []
        for _ in range(num):
            nstep, loss, info = self.do_learn(dat, self.train_collector)

            self.sampled_step += nstep
            info["replay_ratio"] = self.sampled_step / self.collected_step

            self.learn_count += 1
            self.track("learn", info, self.learn_count)
            losses.append(loss)

        return np.mean(losses)

    def test(self, num: int) -> float:
        self.policy.eps = self.eps_test
        self.policy.eval()

        rew = []
        step = []
        act = []
        beg_t = time.time()
        for _ in range(num):
            bat = self.test_collector.collect_trajectory()
            rew.append(sum(bat.rew))
            step.append(len(bat))
            act.append(bat.act)
        cost_t = max(time.time() - beg_t, 1e-6)

        rew_mean = np.mean(rew)
        rew_std = np.std(rew)
        print(f"Epoch #{self.epoch}: {num} test reward={rew_mean:.3f} ± {rew_std:.3f}")

        info = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "dist/rew": np.array(rew, copy=False),
            "step_mean": np.mean(step),
            "step/s": sum(step) / cost_t,
            "dist/step": np.array(step, copy=False),
            "ms/episode": 1000.0 * cost_t / num,
            "dist/act": np.concatenate(act),
        }
        self.track("test", info, self.epoch)
        return rew_mean

    def save(self, rew: float) -> None:
        if rew <= self.best_rew:
            return
        self.policy.train()
        policy = deepcopy(self.policy).to(torch.device("cpu"))
        torch.save(policy.state_dict(), f"{self.log_dir}/{rew:.2f}.pth")

    def track(self, tag: str, info: dict, step: int) -> None:
        if self.writer is None:
            return
        for k, v in info.items():
            if k.startswith("_"):
                continue
            if k.startswith("dist/"):
                self.writer.add_histogram(f"{tag}/{k[5:]}", v, step)
            else:
                self.writer.add_scalar(f"{tag}/{k}", v, step)

    def tqdm_cfg(self):
        return dict(
            ascii=True,
            dynamic_ncols=True,
            desc=f"Epoch #{self.epoch}",
        )


class OffPolicyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_collect(self, nstep_per_iter: int) -> Tuple[Batch, dict]:
        bat = self.train_collector.collect_nstep(nstep_per_iter)
        info = {}
        return bat, info

    def do_learn(self, _: Data, collector: Collector) -> Tuple[int, float, dict]:
        return self.policy.learn(collector=collector)


class OnPolicyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_collect(self) -> Tuple[Batch, dict]:
        bat = self.train_collector.collect_trajectory()
        info = {}
        return bat, info

    def do_learn(self, dat: Data, _: Collector) -> Tuple[int, float, dict]:
        return self.policy.learn(dat=dat)
