import torch

from rlcode.policy.dqn import DQNPolicy
from rlcode.policy.utils.transformed import transformed_h, transformed_h_reverse


class TransformedDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_target_qmax(self, next_obss: torch.FloatTensor) -> torch.FloatTensor:
        targ_qmax = super().compute_target_qmax(next_obss)
        targ_qmax = transformed_h_reverse(targ_qmax)
        return targ_qmax

    def compute_target_qval(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        targ_qval = super().compute_target_qval(next_obss, rews, dones)
        targ_qval = transformed_h(targ_qval)
        return targ_qval
