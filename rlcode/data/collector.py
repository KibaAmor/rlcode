# -*- coding: utf-8 -*-
from typing import Any, Callable, List, Optional, Tuple

import gym
import numpy as np

from rlcode.data.buffer import ReplayBuffer
from rlcode.data.types import Data

Trans = Tuple[np.ndarray, np.ndarray, float, bool, np.ndarray, np.ndarray]


class Collector:
    def __init__(
        self,
        infer: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        env: gym.Env,
        buf: Optional[ReplayBuffer] = None,
        max_episode_step: Optional[int] = None,
    ):
        self.infer = infer
        self.env = env
        self.buf = buf
        self.max_episode_step = max_episode_step

        self.reset = True
        self.obs = None
        self.mask = None
        self.step = 0

    @staticmethod
    def split_obs_mask(obs: Any) -> Tuple[Any, Any]:
        if isinstance(obs, dict):
            return obs["obs"], obs.get("mask")
        else:
            return obs, None

    def _gen(self) -> Trans:
        if self.reset:
            self.reset = False
            self.step = 0
            self.obs, self.mask = self.split_obs_mask(self.env.reset())

        act = self.infer(self.obs, self.mask)
        next_obs, rew, done, _ = self.env.step(act)
        next_obs, mask = self.split_obs_mask(next_obs)

        trans = (self.obs, act, rew, done, next_obs, self.mask)
        if self.buf is not None:
            self.buf.add(*trans)

        self.step += 1
        if done or (self.max_episode_step is not None and self.step >= self.max_episode_step):
            self.reset = True
        else:
            self.obs, self.mask = next_obs, mask

        return trans

    def collect_trajectory(self) -> Data:
        data = []
        while True:
            data.append(self._gen())
            if self.reset:
                break
        return trans2data(data)

    def collect_nstep(self, nstep: int) -> Data:
        data = [self._gen() for _ in range(nstep)]

        return trans2data(data)


def trans2data(trans: List[Trans]) -> Data:
    dat = Data(*zip(*trans))
    if dat.mask[0] is None:
        dat.mask = None
    dat.to_numpy(False)
    dat.act = dat.act.reshape(-1, 1)
    return dat
