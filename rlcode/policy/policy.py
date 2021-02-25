# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch as th
from gym import spaces
from torch import nn


class Policy(ABC, nn.Module):
    def __init__(
        self,
        gamma: float,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        device: Optional[th.device],
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = (
            th.device("cuda" if th.cuda.is_available() else "cpu") if device is None else device
        )
        self.build(**kwargs)

    @abstractmethod
    def build(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs: th.Tensor, mask: Optional[th.Tensor], eps: Optional[float]) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs) -> Tuple[int, float, dict]:
        raise NotImplementedError
