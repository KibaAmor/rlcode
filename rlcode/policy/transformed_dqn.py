import torch

from rlcode.policy.dqn import Batch, DQNPolicy, ExperienceSource, Tuple
from rlcode.policy.utils.transformed import transformed_h, transformed_h_reverse


class TransformedDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        batch, info = super().pre_learn(batch, src)
        batch.rews = transformed_h(batch.rews)
        return batch, info

    def compute_target_q(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        qval_targ = super().compute_target_q(next_obss, rews, dones)
        qval_targ = transformed_h_reverse(qval_targ)
        return qval_targ
