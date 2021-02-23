# -*- coding: utf-8 -*-
from typing import Any, Optional

import numpy as np

from rlcode.data.utils.segtree import SumSegmentTree


class Batch:
    def __init__(
        self,
        obss: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        next_obss: np.ndarray,
        *,
        masks: Optional[np.ndarray] = None,
        indexes: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ):
        self.obss = obss
        self.acts = acts
        self.rews = rews
        self.dones = dones
        self.next_obss = next_obss
        self.masks = masks
        self.indexes = indexes
        self.weights = weights
        self.returns = returns

    def __len__(self):
        return len(self.obss)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, buffer_size: int, batch_size: int):
        assert buffer_size >= batch_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.obss: np.ndarray = None
        self.acts: np.ndarray = None
        self.rews: np.ndarray = None
        self.dones: np.ndarray = None
        self.next_obss: np.ndarray = None
        self.indexes = np.arange(buffer_size)
        self.index = 0
        self.fulled = False

    def __len__(self) -> int:
        return self.buffer_size if self.fulled else self.index

    def _add(self, name: str, value: Any) -> None:
        value = np.array(value)
        batch = getattr(self, name, None)
        if batch is None:
            batch = np.empty((self.buffer_size,) + value.shape)
            setattr(self, name, batch)
        batch[self.index] = value

    def add(
        self,
        obs: Any,
        act: Any,
        rew: float,
        done: bool,
        next_obs: Any,
        mask: Optional[Any] = None,
    ) -> None:
        self._add("obss", obs)
        self._add("acts", act)
        self._add("rews", rew)
        self._add("dones", done)
        self._add("next_obss", next_obs)
        if mask is not None:
            self._add("masks", mask)

        self.index += 1
        if self.index >= self.buffer_size:
            self.fulled = True
            self.index -= self.buffer_size

    def sample(self) -> Batch:
        assert len(self) >= self.batch_size

        indexes = np.random.choice(self.indexes[: len(self)], self.batch_size, False)
        return Batch(
            obss=self.obss[indexes],
            acts=self.acts[indexes],
            rews=self.rews[indexes],
            dones=self.dones[indexes],
            next_obss=self.next_obss[indexes],
            masks=self.masks[indexes] if hasattr(self, "masks") else None,
            indexes=indexes,
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer"""

    def __init__(self, buffer_size: int, batch_size: int, alpha: float, beta: float):
        assert alpha > 0.0 and beta >= 0.0

        super().__init__(buffer_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.eps = np.finfo(np.float).eps.item()

        self.weights = SumSegmentTree(buffer_size)
        self.min_prio = 1.0
        self.max_prio = 1.0

    def add(
        self,
        obs: Any,
        act: Any,
        rew: float,
        done: bool,
        next_obs: Any,
        mask: Optional[Any] = None,
    ):
        self.weights[self.index] = self.max_prio
        super().add(obs, act, rew, done, next_obs, mask)

    def sample(self) -> Batch:
        assert len(self) >= self.batch_size

        scalars = np.random.rand(self.batch_size) * self.weights.reduce()
        indexes = self.weights.get_prefix_sum_index(scalars)
        weights = (self.weights[indexes] / self.min_prio) ** (-self.beta)
        return Batch(
            obss=self.obss[indexes],
            acts=self.acts[indexes],
            rews=self.rews[indexes],
            dones=self.dones[indexes],
            next_obss=self.next_obss[indexes],
            masks=self.masks[indexes] if hasattr(self, "masks") else None,
            indexes=indexes,
            weights=weights,
        )

    def update_weight(self, indexes: np.ndarray, weights: np.ndarray) -> None:
        assert isinstance(indexes, np.ndarray)
        assert isinstance(weights, np.ndarray)

        weights = np.abs(weights) + self.eps
        self.weights[indexes] = weights ** self.alpha
        self.min_prio = min(self.min_prio, weights.min())
        self.max_prio = max(self.max_prio, weights.max())


def create_buffer(
    buffer_size: int, batch_size: int, alpha: float = 0.0, beta: float = 0.0
) -> ReplayBuffer:
    if alpha > 0.0 and beta > 0.0:
        return PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta)
    return ReplayBuffer(buffer_size, batch_size)
