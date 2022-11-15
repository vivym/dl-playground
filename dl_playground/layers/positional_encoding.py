import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.num_channels = num_channels

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, num_channels, 2, dtype=torch.float32) / num_channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: 1d tensor
        assert pos.dim() == 1

        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        return get_emb(sin_inp)


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.num_channels = num_channels

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, num_channels / 2, 2, dtype=torch.float32) / num_channels * 2)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: 2d tensor
        assert pos.dim() == 2 and pos.shape[1] == 2

        pos_x, pos_y = pos[:, 0], pos[:, 1]

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        return torch.cat([emb_x, emb_y], dim=-1)
