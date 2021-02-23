# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty
from functools import reduce
from typing import Optional, Tuple

import torch

from rlcode.data.buffer import Batch
from rlcode.data.experience import ExperienceSource


class Policy(ABC, torch.nn.Module):
    def __init__(self, dist_log_freq: int = 0, device: Optional[torch.device] = None, **kwargs):
        super().__init__()

        self.__dist_log_freq = dist_log_freq
        self.__learn_count = 0

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = device

        self.create_network_optimizer(**kwargs)

    @property
    def learn_count(self) -> int:
        return self.__learn_count

    @property
    def can_log_dist(self) -> bool:
        return self.__dist_log_freq > 0 and self.__learn_count % self.__dist_log_freq == 0

    @property
    def device(self) -> torch.device:
        return self.__device

    @abstractproperty
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractproperty
    def network(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_network_optimizer(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, obss: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def pre_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        raise NotImplementedError

    @abstractmethod
    def do_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        raise NotImplementedError

    def post_learn(self, batch: Batch, src: ExperienceSource) -> dict:
        info = {}
        # for param_group in self.optimizer.param_groups:
        #     info["lr"] = param_group["lr"]
        #     break

        if self.can_log_dist:
            weight = reduce(
                lambda x, y: torch.cat((x.reshape(-1), y.reshape(-1))),
                map(lambda x: x.data, self.network.parameters()),
            )
            grad = reduce(
                lambda x, y: torch.cat((x.reshape(-1), y.reshape(-1))),
                map(lambda x: x.grad, self.network.parameters()),
            )
            info["dist/weight"] = weight
            info["dist/grad"] = grad
            # for name, param in self.network.named_parameters():
            #     info[f"dist/network/{name}"] = param
            #     info[f"dist/grad/{name}"] = param.grad

        self.__learn_count += 1
        return info

    def learn(self, batch: Batch, src: ExperienceSource) -> dict:
        batch, pre_info = self.pre_learn(batch, src)
        batch, do_info = self.do_learn(batch, src)
        post_info = self.post_learn(batch, src)
        info = dict(**pre_info, **do_info, **post_info)
        return info
