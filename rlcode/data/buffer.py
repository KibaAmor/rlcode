# -*- coding: utf-8 -*-
import operator
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from gym import spaces

from rlcode.data.types import Batch, PrioBatch

# region basic types and replay buffer


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        save_next_obs: bool = False,
        has_mask: bool = False,
        always_copy: bool = False,
    ) -> None:
        assert batch_size > 0 and buffer_size >= batch_size
        obs_shape = get_obs_shape(obs_space)
        act_dim = get_act_dim(act_space)

        self.obs = np.zeros((buffer_size,) + obs_shape, dtype=float)
        self.act = np.zeros((buffer_size, act_dim), dtype=act_space.dtype)
        self.rew = np.zeros((buffer_size,), dtype=float)
        self.done = np.zeros((buffer_size,), dtype=bool)
        self.next_obs = np.zeros((buffer_size,) + obs_shape, dtype=float) if save_next_obs else None
        self.mask = np.zeros((buffer_size, act_dim), dtype=bool) if has_mask else None

        self.always_copy = always_copy
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        next_obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.obs[self.pos] = np.array(obs, copy=self.always_copy)
        self.act[self.pos] = np.array(act, copy=self.always_copy)
        self.rew[self.pos] = rew
        self.done[self.pos] = done
        if self.next_obs is not None:
            self.next_obs[self.pos] = np.array(next_obs, copy=self.always_copy)
        if self.mask is not None:
            self.mask[self.pos] = np.array(mask, copy=self.always_copy)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos -= self.buffer_size

    def sample(self, has_next_obs: bool) -> Batch:
        index = np.random.choice(len(self), size=self.batch_size, replace=False)
        return self.get(index, has_next_obs)

    def get_all(self, has_next_obs: bool) -> Batch:
        index = np.arange(len(self))
        return self.get(index, has_next_obs)

    def get(self, index: np.ndarray, has_next_obs: bool) -> Batch:
        if has_next_obs:
            next_obs = (
                self.next_obs[index]
                if self.next_obs is not None
                else self.obs[(index + 1) % len(self)]
            )
        else:
            next_obs = None
        mask = self.mask[index] if self.mask is not None else None

        return Batch(
            obs=self.obs[index],
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            next_obs=next_obs,
            mask=mask,
            index=index,
        )


class PrioReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer"""

    def __init__(
        self,
        alpha: float,
        beta: float,
        buffer_size: int,
        batch_size: int,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        save_next_obs: bool = False,
        has_mask: bool = False,
        always_copy: bool = False,
    ) -> None:
        super().__init__(
            buffer_size,
            batch_size,
            obs_space,
            act_space,
            save_next_obs,
            has_mask,
            always_copy,
        )

        self.weight = SumSegmentTree(buffer_size)
        self.min_prio = 1.0
        self.max_prio = 1.0

        self.alpha = alpha
        self.beta = beta
        self.eps = np.finfo(float).eps.item()

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        next_obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.weight[self.pos] = self.max_prio
        super().add(obs, act, rew, done, next_obs, mask)

    def sample(self, has_next_obs: bool) -> PrioBatch:
        scalar = np.random.rand(self.batch_size) * self.weight.reduce()
        index = self.weight.get_prefix_sum_index(scalar)
        return self.get(index, has_next_obs)

    def get_all(self, has_next_obs: bool) -> PrioBatch:
        index = np.arange(len(self))
        return self.get(index, has_next_obs)

    def get(self, index: np.ndarray, has_next_obs: bool) -> PrioBatch:
        bat = super().get(index, has_next_obs)
        weight = (self.weight[index] / self.min_prio) ** (-self.beta)
        return PrioBatch(**bat.__dict__, weight=weight)

    def update(self, index: np.ndarray, weight: np.ndarray) -> None:
        weight = np.abs(weight) + self.eps
        self.weight[index] = weight ** self.alpha
        self.min_prio = min(self.min_prio, weight.min())
        self.max_prio = max(self.max_prio, weight.max())


# endregion


def get_obs_shape(obs_space: spaces.Space) -> Tuple[int, ...]:
    if isinstance(obs_space, spaces.Discrete):
        return (1,)
    else:
        return obs_space.shape


def get_act_dim(act_space: spaces.Space) -> int:
    if isinstance(act_space, spaces.Discrete):
        return 1
    else:
        return int(np.prod(act_space.shape))


def create_buffer(
    alpha: float,
    beta: float,
    buffer_size: int,
    batch_size: int,
    obs_space: spaces.Space,
    act_space: spaces.Space,
    save_next_obs: bool = False,
    has_mask: bool = False,
    always_copy: bool = False,
) -> Union[ReplayBuffer, PrioReplayBuffer]:
    common_args = [
        buffer_size,
        batch_size,
        obs_space,
        act_space,
        save_next_obs,
        has_mask,
        always_copy,
    ]
    if alpha > 0.0 and beta > 0.0:
        return PrioReplayBuffer(alpha, beta, *common_args)
    return ReplayBuffer(*common_args)


# region SegmentTree and SumSegmentTree


class SegmentTree:
    """Segment tree"""

    def __init__(self, size: int, operation: Callable[[Any, Any], Any]):
        capacity = 1
        while capacity < size:
            capacity <<= 1

        self._size = size
        self._capacity = capacity
        self._operation = operation
        self._value = np.zeros([capacity * 2], dtype=float)

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        beg, end = self._capacity, self._capacity + self._size
        return iter(self._value[beg:end])

    def __getitem__(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        assert isinstance(index, (int, np.ndarray))
        assert np.all(0 <= index) and np.all(index < len(self))

        return self._value[self._capacity + index]

    def __setitem__(self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]) -> None:
        assert np.all(0 <= index) and np.all(index < len(self))

        if not isinstance(index, np.ndarray):
            index = np.array([index])
            value = np.array([value])

        index = index + self._capacity  # DO NOT MODIFY INPUT
        self._value[index] = value
        while index[0] > 1:
            self._value[index >> 1] = self._operation(self._value[index], self._value[index ^ 1])
            index >>= 1

    def reduce(self, beg: int = 0, end: Optional[int] = None) -> float:
        """Return sum(self[beg:end])"""

        if beg == 0 and end is None:
            return self._value[1]

        if end is None:
            end = self._size
        elif end < 0:
            end += self._size
        beg, end = beg + self._capacity, end + self._capacity

        result = 0.0
        while beg < end:
            if beg & 1:
                result += self._value[beg]
                beg += 1
            beg >>= 1
            if end & 1:
                end -= 1
                result += self._value[end]
            end >>= 1

        return result


class SumSegmentTree(SegmentTree):
    def __init__(self, size: int):
        super().__init__(size, operator.add)

    def get_prefix_sum_index(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Return index which sum(self[:index-1]) < value <= sum(self[:index])"""

        single = False
        if not isinstance(value, np.ndarray):
            single = True
            value = np.array([value], dtype=float)
        assert np.all(0.0 <= value) and np.all(value <= self._value[1])

        value = value.copy()  # DO NOT MODIFY INPUT
        index = np.ones_like(value, dtype=np.int)
        while index[0] < self._capacity:
            index <<= 1
            left_value = self._value[index]
            direction = left_value < value
            value -= left_value * direction
            index += direction
        index -= self._capacity

        return index.item() if single else index


# endregion
