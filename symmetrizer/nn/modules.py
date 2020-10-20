import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from symmetrizer.groups.groups import MatrixRepresentation
from symmetrizer.ops.ops import get_basis, get_coeffs, get_invariant_basis, \
    compute_gain


class BasisLayer(torch.nn.Module):
    """
    Linear layer forward pass
    """
    def forward(self, x):
        """
        Normal forward pass, using weights formed by the basis
        and corresponding coefficients
        """
        self.W = torch.sum(self.basis*self.coeffs, 0)

        x = x[:, None, None, :, :]
        self.W = self.W[None, :, :, :, :]
        wx = self.W * x
        out = torch.sum(wx, [-2, -1])
        if self.has_bias:
            self.b = torch.sum(self.basis_bias*self.coeffs_bias, 0)
            return out + self.b
        else:
            return out


class BasisLinear(BasisLayer):
    """
    Group-equivariant linear layer
    """
    def __init__(self, channels_in, channels_out, group, bias=True,
                 n_samples=4096, basis="equivariant", gain_type="xavier",
                 bias_init=False):
        """
        """
        super().__init__()

        self.group = group
        self.space = basis
        self.repr_size_in = group.repr_size_in
        self.repr_size_out = group.repr_size_out
        self.channels_in = channels_in
        self.channels_out = channels_out

        ### Getting Basis ###
        size = (n_samples, self.repr_size_out, self.repr_size_in)
        new_size = [1, self.repr_size_out, 1, self.repr_size_in]
        basis, self.rank = get_basis(size, group, new_size, space=self.space)
        self.register_buffer("basis", basis)

        gain = compute_gain(gain_type, self.rank, self.channels_in,
                            self.channels_out, self.repr_size_in,
                            self.repr_size_out)

        ### Getting Coefficients ###
        size = [self.rank, self.channels_out, 1, self.channels_in, 1]
        self.coeffs = get_coeffs(size, gain)

        self.has_bias = False
        if bias:
            self.has_bias = True
            size = [n_samples, self.repr_size_out, 1]
            new_size = [1, self.repr_size_out]
            basis_bias, self.rank_bias = get_invariant_basis(size, group,
                                                             new_size,
                                                             space=self.space)

            self.register_buffer("basis_bias", basis_bias)
            if not bias_init:
                gain = 1.
            else:
                gain = compute_gain(gain_type, self.rank_bias,
                                    self.channels_in, self.channels_out,
                                    self.repr_size_in, self.repr_size_out)

            size = [self.rank_bias, self.channels_out, 1]
            self.coeffs_bias = get_coeffs(size, gain=gain)


    def __repr__(self):
        string = f"{self.space} Linear({self.repr_size_in}, "
        string += f"{self.channels_in}, {self.repr_size_out}, "
        string += f"{self.channels_out}), bias={self.has_bias})"
        return string
