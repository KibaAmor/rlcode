# -*- coding: utf-8 -*-
from abc import abstractproperty
from copy import deepcopy
from typing import Optional, Tuple, cast

import numpy as np
import torch as th

from rlcode.data.buffer import PrioBatch, PrioReplayBuffer
from rlcode.data.collector import Collector
from rlcode.policy.policy import Policy


class DQNPolicy(Policy):
    def __init__(self, target_update_freq: int, *args, **kwargs):
        self.tuf = target_update_freq
        self.count = 0
        self.eps = 0
        super().__init__(*args, **kwargs)

        if self.tuf > 0:
            self.__target_net = deepcopy(self.net)
            self.__target_net.eval()

    @abstractproperty
    def optim(self) -> th.optim.Optimizer:
        raise NotImplementedError

    @abstractproperty
    def net(self) -> th.nn.Module:
        raise NotImplementedError

    def calc_target_qval(self, next_obs: th.Tensor, rew: th.Tensor, done: th.Tensor) -> th.Tensor:
        with th.no_grad():
            if self.tuf > 0:
                act = self.net(next_obs).argmax(1).unsqueeze_(1)
                logits = self.__target_net(next_obs).gather(1, act)
            else:
                logits = self.net(next_obs).max(1, keepdim=True)[0]
        qval = self.logits2qval(logits)
        return rew + (1 - done) * self.gamma * qval

    def logits2qval(self, logits: th.Tensor) -> th.Tensor:
        return logits

    def infer(self, obs: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        obs = th.as_tensor(obs, dtype=th.float32, device=self.device).unsqueeze_(0)
        mask = (
            th.as_tensor(mask, dtype=bool, device=self.device).unsqueeze_(0)
            if mask is not None
            else None
        )
        with th.no_grad():
            qval = self(obs, mask, None).squeeze_(0)
        return qval.argmax(0).cpu().numpy()

    def forward(self, obs: th.Tensor, mask: Optional[th.Tensor], eps: Optional[float]) -> th.Tensor:
        qval = self.net(obs)

        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0.0):
            prob = th.rand(qval.shape[0])
            index = th.where(prob < eps)[0]
            if index.shape[0] > 0:
                qval[index] = th.rand(
                    index.shape[:1] + qval.shape[1:], dtype=qval.dtype, device=self.device
                )

        if mask is not None:
            qval.masked_fill_(~mask, -np.inf)

        return qval

    def learn(self, collector: Collector) -> Tuple[int, float, dict]:
        if self.tuf > 0 and self.count % self.tuf == 0:
            self.__target_net.load_state_dict(self.net.state_dict())
        self.count += 1

        buf = collector.buf
        bat = buf.sample(True).to_tensor(self.device, False)

        predict_qval = self.net(bat.obs).gather(1, bat.act)
        target_qval = self.calc_target_qval(bat.next_obs, bat.rew, bat.done)

        td_err = predict_qval - target_qval
        if isinstance(collector, PrioReplayBuffer) and isinstance(bat, PrioBatch):
            buf = cast(PrioReplayBuffer, buf)
            bat = cast(PrioBatch, bat)

            loss = (td_err.pow(2) * bat.weight).mean()

            buf.update(bat.index, td_err)
        else:
            loss = td_err.pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return len(bat), loss.item(), {}
