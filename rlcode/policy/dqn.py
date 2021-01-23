from abc import abstractmethod, abstractproperty
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch

from rlcode.data.buffer import Batch, PrioritizedReplayBuffer, ReplayBuffer
from rlcode.data.experience import ExperienceSource
from rlcode.policy.policy import Policy


class DQNPolicy(Policy):
    def __init__(
        self,
        gamma: float,
        tau: float = 1.0,
        target_update_freq: int = 0,
        dist_log_freq: int = 0,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(device)
        self._gamma = gamma
        self._tau = tau
        self._target_update_freq = target_update_freq
        self._dist_log_freq = dist_log_freq

        self._learn_count = 0
        self.eps = 0.0

        self.create_network_optimizer(**kwargs)
        if target_update_freq > 0:
            self._target_network = deepcopy(self.network).to(self.device)
            self._target_network.load_state_dict(self.network.state_dict())
            self._target_network.eval()
            for param in self._target_network.parameters():
                param.requires_grad = False

    @abstractmethod
    def create_network_optimizer(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractproperty
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractproperty
    def network(self) -> torch.nn.Module:
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self, obss: torch.tensor, masks: Optional[torch.tensor] = None
    ) -> torch.tensor:
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

        info = {}
        if batch.weights is not None:
            info["weights_mean"] = np.mean(batch.weights)
            info["weights_std"] = np.std(batch.weights)
            info["weights_min"] = np.min(batch.weights)
            info["weights_max"] = np.max(batch.weights)

        batch.to_tensor(self.device)
        return batch, info

    def do_learn(self, batch: Batch, src: ExperienceSource) -> Tuple[Batch, dict]:
        obss = batch.obss
        acts = batch.acts.unsqueeze(1)
        rews = batch.rews
        dones = batch.dones
        next_obss = batch.next_obss
        weights = batch.weights if batch.weights is not None else 1.0

        qval_pred = self.network(obss).gather(1, acts).squeeze()
        qval_targ = self.compute_target_q(next_obss, rews, dones)
        td_err = qval_pred - qval_targ

        loss = (td_err.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {"loss": loss.item()}
        for param_group in self.optimizer.param_groups:
            info["lr"] = param_group["lr"]
            break

        self._learn_count += 1
        batch.weights = td_err
        return batch, info

    def post_learn(self, batch: Batch, src: ExperienceSource) -> dict:
        td_err = batch.weights
        info = {
            "err_mean": td_err.mean().item(),
            "err_std": td_err.std().item(),
            "err_min": td_err.min().item(),
            "err_max": td_err.max().item(),
        }

        buffer: PrioritizedReplayBuffer = getattr(src, "buffer", None)
        if isinstance(buffer, PrioritizedReplayBuffer):
            buffer.update_weight(batch.indexes, td_err.cpu().data.numpy())

        if self._dist_log_freq > 0 and self._learn_count % self._dist_log_freq == 0:
            for name, param in self.network.named_parameters():
                info[f"dist/network/{name}"] = param
                info[f"dist/grad/{name}"] = param.grad

        if (
            self._target_update_freq > 0
            and self._learn_count % self._target_update_freq == 0
        ):
            self._update_target_network()

        return info

    def _update_target_network(self) -> None:
        if np.isclose(self._tau, 0):
            self._target_network.load_state_dict(self.network.state_dict())
        else:
            pairs = zip(self.network.parameters(), self._target_network.parameters())
            for src, dst in pairs:
                dst.data.copy_(self._tau * src.data + (1.0 - self._tau) * dst.data)

    def compute_target_q(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            if self._target_update_freq > 0:
                acts = self.network(next_obss).argmax(-1).unsqueeze(1)
                qval_max = self._target_network(next_obss).gather(1, acts).squeeze()
            else:
                qval_max = self.network(next_obss).max(-1)[0]
        qval_targ = rews + (1 - dones) * self._gamma * qval_max
        return qval_targ
