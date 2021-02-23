# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import gym
import torch

from rlcode.data.buffer import Batch, ReplayBuffer


class Experience:
    def __init__(
        self,
        obs: Any,
        act: Any,
        rew: float,
        done: bool,
        next_obs: Any,
        mask: Any,
    ):
        self.obs = obs
        self.act = act
        self.rew = rew
        self.done = done
        self.next_obs = next_obs
        self.mask = mask

    def __iter__(self):
        return (
            v
            for v in (
                self.obs,
                self.act,
                self.rew,
                self.done,
                self.next_obs,
                self.mask,
            )
        )

    def __repr__(self):
        return "Experience(obs={}, act={}, rew={}, done={}, next_obs={}, mask={})".format(
            self.obs, self.act, self.rew, self.done, self.next_obs, self.mask
        )

    def to_batch(self) -> Batch:
        return Batch(
            obss=self.obs,
            acts=self.act,
            rews=self.rew,
            dones=self.done,
            next_obss=self.next_obs,
            masks=self.mask,
        )


class ExperienceSource(ABC):
    def __init__(self, policy, env: gym.Env, max_episode_step: Optional[int] = None):
        self._policy = policy
        self._env = env
        self._max_episode_step = max_episode_step

        self._reset = True
        self._obs = None
        self._mask = None
        self._steps = 0

    def _generate(self) -> Experience:
        if self._reset:
            self._reset = False
            self._obs, self._mask = self.split_obs_mask(self._env.reset())
            self._steps = 0

        obss = torch.FloatTensor([self._obs])
        masks = torch.BoolTensor([self._mask]) if self._mask is not None else None
        act = self._policy(obss, masks).item()

        next_obs, rew, done, _ = self._env.step(act)
        nobs, nmask = self.split_obs_mask(next_obs)

        exp = Experience(self._obs, act, rew, done, nobs, self._mask)

        self._steps += 1
        if done or (self._max_episode_step is not None and self._steps >= self._max_episode_step):
            self._reset = True
        else:
            self._obs, self._mask = nobs, nmask

        return exp

    @staticmethod
    def split_obs_mask(obs: Any) -> Tuple[Any, Any]:
        if isinstance(obs, dict):
            return obs["obs"], obs.get("mask")
        else:
            return obs, None

    @abstractmethod
    def collect(self, **kwargs) -> Batch:
        raise NotImplementedError


class NStepExperienceSource(ExperienceSource):
    def __init__(
        self,
        policy,
        env: gym.Env,
        buffer: Optional[ReplayBuffer],
        nstep: int,
        max_episode_step: Optional[int] = None,
    ):
        super().__init__(policy, env, max_episode_step)
        self._nstep = nstep
        self._buffer = buffer

    @property
    def buffer(self) -> ReplayBuffer:
        return self._buffer

    def collect(self, **kwargs) -> Batch:
        exps = [self._generate() for _ in range(self._nstep)]
        if self._buffer is not None:
            for exp in exps:
                self._buffer.add(*exp)

        exps = Experience(*(zip(*exps)))
        return exps.to_batch()


class EpisodeExperienceSource(ExperienceSource):
    def __init__(self, policy, env: gym.Env, max_episode_step: Optional[int] = None):
        super().__init__(policy, env, max_episode_step)

    def collect(self, **kwargs) -> Batch:
        exps = []
        while True:
            exps.append(self._generate())
            if self._reset:
                break

        exps = Experience(*(zip(*exps)))
        return exps.to_batch()
