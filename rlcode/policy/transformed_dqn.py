# -*- coding: utf-8 -*-
import torch as th

from rlcode.policy.dqn import DQNPolicy
from rlcode.policy.utils.transformed import transformed_h, transformed_h_reverse


class TransformedDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_target_qval(self, next_obs: th.Tensor, rew: th.Tensor, done: th.Tensor) -> th.Tensor:
        qval = super().calc_target_qval(next_obs, rew, done)
        return transformed_h(qval)

    def logits2qval(self, logits: th.Tensor) -> th.Tensor:
        return transformed_h_reverse(logits)
