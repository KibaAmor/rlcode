from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch as th


@dataclass
class Data:
    obs: Union[th.Tensor, np.ndarray]
    act: Union[th.Tensor, np.ndarray]
    rew: Union[th.Tensor, np.ndarray]
    done: Union[th.Tensor, np.ndarray]
    next_obs: Optional[Union[th.Tensor, np.ndarray]]
    mask: Optional[Union[th.Tensor, np.ndarray]]

    def __len__(self):
        return len(self.obs)

    def to_tensor(self, device: Union[th.device, str], copy: bool) -> Data:
        self.obs = to_tensor(self.obs, th.float32, device, copy)
        self.act = to_tensor(self.act, None, device, copy)
        self.rew = to_tensor(self.rew, th.float32, device, copy)
        self.done = to_tensor(self.done, int, device, copy)
        self.next_obs = to_tensor(self.next_obs, th.float32, device, copy)
        self.mask = to_tensor(self.mask, bool, device, copy)
        return self

    def to_numpy(self, copy: bool) -> Data:
        self.obs = to_numpy(self.obs, np.float32, copy)
        self.act = to_numpy(self.act, None, copy)
        self.rew = to_numpy(self.rew, np.float32, copy)
        self.done = to_numpy(self.done, int, copy)
        self.next_obs = to_numpy(self.next_obs, np.float32, copy)
        self.mask = to_numpy(self.mask, bool, copy)
        return self


@dataclass
class Batch(Data):
    index: Union[th.Tensor, np.ndarray]

    def to_tensor(self, device: Union[th.device, str], copy: bool) -> Batch:
        super().to_tensor(device, copy)
        self.index = to_tensor(self.index, int, device, copy)
        return self

    def to_numpy(self, copy: bool) -> Batch:
        super().to_numpy(copy)
        self.index = to_numpy(self.index, int, copy)
        return self


@dataclass
class PrioBatch(Batch):
    weight: Union[th.Tensor, np.ndarray]

    def to_tensor(self, device: Union[th.device, str], copy: bool) -> PrioBatch:
        super().to_tensor(device, copy)
        self.weight = to_tensor(self.weight, th.float32, device, copy)
        return self

    def to_numpy(self, copy: bool) -> Batch:
        super().to_numpy(copy)
        self.weight = to_numpy(self.weight, np.float32, copy)
        return self


def to_tensor(
    obj: Any, dtype: Any, device: Union[th.device, str], copy: bool
) -> Optional[th.Tensor]:
    if obj is None:
        return None
    return (
        th.tensor(obj, dtype=dtype, device=device)
        if copy
        else th.as_tensor(obj, dtype=dtype, device=device)
    )


def to_numpy(obj: Any, dtype: Any, copy: bool) -> Optional[np.ndarray]:
    if obj is None:
        return None
    return np.array(obj, dtype=dtype, copy=copy)
