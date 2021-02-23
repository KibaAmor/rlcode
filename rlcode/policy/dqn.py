# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch

from rlcode.data.buffer import Batch, PrioritizedReplayBuffer, ReplayBuffer
from rlcode.data.experience import ExperienceSource
from rlcode.policy.policy import Policy


class DQNPolicy(Policy):
    def __init__(self, gamma: float, tau: float = 1.0, target_update_freq: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._gamma = gamma
        self._tau = tau
        self._target_update_freq = target_update_freq
        self.eps = 0.0

        if target_update_freq > 0:
            self._target_network = deepcopy(self.network).to(self.device)
            self._target_network.load_state_dict(self.network.state_dict())
            self._target_network.eval()
            for param in self._target_network.parameters():
                param.requires_grad = False

    def forward(self, obss: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            qvals = self.network(obss.to(self.device))

        if not np.isclose(self.eps, 0.0):
            for i in range(len(qvals)):
                if np.random.rand() < self.eps:
                    torch.rand(qvals[i].shape, device=self.device, out=qvals[i])

        if masks is not None:
            masks = torch.BoolTensor(masks).to(self.device)
            qvals.masked_fill_(~masks, -np.inf)

        acts = qvals.argmax(-1)
        return acts

    def pre_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        buffer: ReplayBuffer = getattr(src, "buffer", None)
        if isinstance(buffer, ReplayBuffer):
            batch = buffer.sample()

        info = {"_batch_size": len(batch)}
        if batch.weights is not None and self.can_log_dist:
            info["dist/prioritized_weight"] = batch.weights

        batch.obss = torch.FloatTensor(batch.obss).to(self.device)
        batch.acts = torch.LongTensor(batch.acts).to(self.device)
        batch.rews = torch.FloatTensor(batch.rews).to(self.device)
        batch.dones = torch.LongTensor(batch.dones).to(self.device)
        batch.next_obss = torch.FloatTensor(batch.next_obss).to(self.device)
        if batch.weights is not None:
            batch.weights = torch.FloatTensor(batch.weights).to(self.device)
        return batch, info

    def do_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        obss = batch.obss
        acts = batch.acts.unsqueeze(1)
        rews = batch.rews
        dones = batch.dones
        next_obss = batch.next_obss
        weights = batch.weights if batch.weights is not None else 1.0

        pred_qval = self.network(obss).gather(1, acts).squeeze()
        targ_qval = self.compute_target_qval(next_obss, rews, dones)
        td_err = pred_qval - targ_qval
        batch.weights = td_err

        loss = (td_err.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {
            "loss": loss.item(),
        }
        if self.can_log_dist:
            info["dist/qval"] = pred_qval
            info["dist/td_err"] = td_err

        return batch, info

    def post_learn(self, batch: Batch, src: ExperienceSource) -> dict:
        buffer: PrioritizedReplayBuffer = getattr(src, "buffer", None)
        if isinstance(buffer, PrioritizedReplayBuffer):
            td_err = batch.weights
            buffer.update_weight(batch.indexes, td_err.cpu().data.numpy())

        if self._target_update_freq > 0 and self.learn_count % self._target_update_freq == 0:
            self._update_target_network()

        return super().post_learn(batch, src)

    def _update_target_network(self) -> None:
        if np.isclose(self._tau, 1.0):
            self._target_network.load_state_dict(self.network.state_dict())
        else:
            pairs = zip(self.network.parameters(), self._target_network.parameters())
            for src, dst in pairs:
                dst.data.copy_(self._tau * src.data + (1.0 - self._tau) * dst.data)

    def compute_target_qmax(self, next_obss: torch.FloatTensor) -> torch.FloatTensor:
        if self._target_update_freq > 0:
            acts = self.network(next_obss).argmax(-1).unsqueeze(1)
            targ_qmax = self._target_network(next_obss).gather(1, acts).squeeze()
        else:
            targ_qmax = self.network(next_obss).max(-1)[0]
        return targ_qmax

    def compute_target_qval(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            targ_qmax = self.compute_target_qmax(next_obss)
        return rews + (1 - dones) * self._gamma * targ_qmax
