from typing import Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from rlcode.data.buffer import Batch
from rlcode.data.experience import ExperienceSource
from rlcode.policy.policy import Policy


class PGPolicy(Policy):
    def __init__(
        self,
        gamma: float,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._gamma = gamma
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def forward(
        self, obss: torch.tensor, masks: Optional[torch.tensor] = None
    ) -> torch.tensor:
        with torch.no_grad():
            probs = self.network(obss.to(self.device))

        if not np.isclose(self.eps, 0.0):
            for i in range(len(probs)):
                if np.random.rand() < self.eps:
                    torch.rand(probs[i].shape, device=self.device, out=probs[i])

        if masks is not None:
            masks = torch.BoolTensor(masks).to(self.device)
            probs.masked_fill_(~masks, 0)

        dists = Categorical(probs)
        acts = dists.sample()
        return acts

    def pre_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        batch.to_tensor(self.device)

        for i in range(len(batch.rews) - 2, -1, -1):
            batch.rews[i] += self._gamma * batch.rews[i + 1]

        rew_std = batch.rews.std()
        if np.isclose(rew_std.item(), 0, 1e-3):
            batch.rews = (batch.rews - batch.rews.mean()) / rew_std

        return batch, {}

    def do_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        dataset = TensorDataset(batch.obss, batch.acts, batch.rews)
        loader = DataLoader(
            dataset, self._batch_size, self._shuffle, drop_last=self._drop_last
        )

        losses = []
        for obss, acts, rews in loader:
            probs = self.network(obss)
            dists = Categorical(probs)
            loss = -(dists.log_prob(acts) * rews).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        info = {
            "loss": np.mean(losses),
        }
        for param_group in self.optimizer.param_groups:
            info["lr"] = param_group["lr"]
            break

        return batch, info