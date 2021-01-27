from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

import torch

from rlcode.data.buffer import Batch
from rlcode.data.experience import ExperienceSource


class Policy(ABC, torch.nn.Module):
    def __init__(self, device: Optional[torch.device] = None, **kwargs):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.create_network_optimizer(**kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

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
    def forward(
        self, obss: torch.tensor, masks: Optional[torch.tensor] = None
    ) -> torch.tensor:
        raise NotImplementedError

    @abstractmethod
    def pre_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        raise NotImplementedError

    @abstractmethod
    def do_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        raise NotImplementedError

    def post_learn(self, batch: Batch, src: ExperienceSource) -> dict:
        return {}

    def learn(self, batch: Batch, src: ExperienceSource) -> dict:
        batch, pre_info = self.pre_learn(batch, src)
        batch, do_info = self.do_learn(batch, src)
        post_info = self.post_learn(batch, src)
        info = dict(**pre_info, **do_info, **post_info)
        return info
