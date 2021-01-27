from typing import Callable, List, Optional, Union

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
        adv_layer_num: int = 0,
        val_layer_num: int = 0,
        activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        if adv_layer_num > 0 and val_layer_num > 0:
            self._dueling = True
            self._com = mlp([obs_n] + [hidden_size] * layer_num, activation, activation)
            self._adv = mlp([hidden_size] * adv_layer_num + [act_n], activation)
            self._val = mlp([hidden_size] * val_layer_num + [1], activation)
        else:
            self._dueling = False
            self._net = mlp([obs_n] + [hidden_size] * layer_num + [act_n], activation)

    def forward(self, obs):
        if not self._dueling:
            return self._net(obs)

        com = self._com(obs)
        adv = self._adv(com)
        val = self._val(com)
        return val - adv.mean() + adv


class PNet(nn.Module):
    def __init__(
        self,
        obs_n: int,
        act_n: int,
        layer_num: int,
        hidden_size: int,
        activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        self._net = mlp(
            [obs_n] + [hidden_size] * layer_num + [act_n],
            activation,
            lambda: nn.Softmax(-1),
        )

    def forward(self, obs):
        return self._net(obs)
