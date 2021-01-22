from typing import Callable, List, Optional, Tuple

from torch import nn


def mlp(
    sizes: List[int],
    activation: Optional[Callable[[], nn.Module]] = None,
    output_activation: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        act = activation if i < len(sizes) - 2 else output_activation
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(
        self,
        obs_n: int,
        act_n: int,
        layer_num: int,
        hidden_size: int,
        dueling: Optional[Tuple[int, int]] = None,
        activation: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        self._dueling = dueling

        if dueling is None:
            self.net = mlp([obs_n] + [hidden_size] * layer_num + [act_n], activation)
        else:
            self.com = mlp([obs_n] + [hidden_size] * layer_num, activation, activation)
            self.adv = mlp([hidden_size] * dueling[0] + [act_n], activation)
            self.val = mlp([hidden_size] * dueling[1] + [1], activation)

    def forward(self, obs):
        if self._dueling is None:
            return self.net(obs)

        com = self.com(obs)
        adv = self.adv(com)
        val = self.val(com)
        return val - adv.mean() + adv
