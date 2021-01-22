import operator
from typing import Any, Callable, Optional, Union

import numpy as np


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

    def __setitem__(
        self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        assert np.all(0 <= index) and np.all(index < len(self))

        if not isinstance(index, np.ndarray):
            index = np.array([index])
            value = np.array([value])

        index = index + self._capacity  # DO NOT MODIFY INPUT
        self._value[index] = value
        while index[0] > 1:
            self._value[index >> 1] = self._operation(
                self._value[index], self._value[index ^ 1]
            )
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

    def get_prefix_sum_index(
        self, value: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
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
