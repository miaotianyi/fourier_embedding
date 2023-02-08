import math
import torch
from torch import nn

from einops import einsum


class Depthwise1d(nn.Module):
    def __init__(self,
                 num_channels: int,
                 h_in: int,
                 h_out: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        """
        Use n distinct linear (MLP) layers to separately transform n channels,
        such that each channel changes from length `h_in` to length `h_out`.

        Shape:
            - Input: `(*, C, H_in)`, where `∗` means any number of dimensions including none.
            - Output: `(*, C, H_out)`, where all but the last dimension have the same shape
            as the input.

        Attributes:
            - weight: the learnable weights of the module of shape `(C, H_out, H_in)`,
            initialized from `Uniform(-1/sqrt(H_in), 1/sqrt(H_in))`.
            - bias: the learnable bias of the module of shape (C, H_out)`,
            initialized from `Uniform(-1/sqrt(H_in), 1/sqrt(H_in))`.

        This is a generalization of feature-tokenizer layer in
        FT-Transformer (https://arxiv.org/abs/2106.11959),
        which uses different linear layers to embed each numerical feature
        to a vector, mapping from `(N, C)` to `(N, C, H_out)`.
        We generalize it to allow input features that are already embedding vectors,
        mapping from `(N, C, H_in)` to `(N, C, H_out)`.
        In the original paper, feature tokenizer is only applied to numerical features,
        but we make this layer modular, so it's left for users to decide
        which architecture to choose (e.g. apply after categorical `nn.Embedding`).

        The idea is very similar to depthwise convolution
        (convolve each channel with a different kernel respectively, operating on space),
        which is first introduced by MobileNet (https://arxiv.org/abs/1704.04861)
        and typically followed by a 1x1 convolution block that operates on channels.

        I might change the implementation when einops starts to support ... in EinMix.

        Parameters
        ----------
        num_channels : int
            The number of channels `C` in input tensor of shape `[N, C, H_in]`.

        h_in : int
            The embedding size of the input features. 1 if there's no embedding.

        h_out : int
            The output embedding size after the layer.

        bias : bool, default: True
            Whether the linear layer has bias.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.h_in = h_in
        self.h_out = h_out
        # this weight is shaped like this to preserve similarity to the Conv1d implementation
        self.weight = nn.Parameter(torch.empty((num_channels, h_out, h_in), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_channels, h_out), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # we mimic the initialization of nn.Linear:
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        # However, calculation of fan_in and fan_out is not supported for depthwise MLP weights
        # So we initialize manually instead.
        bound = 1 / math.sqrt(self.h_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            if self.h_in == 0:
                bound = 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _check_input_dim(self, x):
        if x.ndim < 2:
            raise ValueError(f"Expected input of shape [*, C, H_in] to have at least 2 dimensions, got {list(x.shape)}")
        if x.shape[-2] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {list(x.shape)}")
        if x.shape[-1] != self.h_in:
            raise ValueError(f"Expected H_in={self.h_in}, got {list(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        x = einsum(x, self.weight, "... c h_in, c h_out h_in -> ... c h_out")
        if self.bias is not None:
            # x has shape [..., c, h_out], bias has shape [c, h_out]
            x = x + self.bias
        return x


# This is an old implementation
# The new implementation using einsum is equivalent
# (including random parameter initialization given the same seed)
# unless some unknown behaviors exist for Conv1d during backprop.
# Experimentally, they are also the same.
# I use einsum because it is easier to understand.
#
# from einops import rearrange, pack, unpack,
# class Depthwise1d(nn.Module):
#     def __init__(self, num_channels, h_in, h_out, bias=True,
#                  device=None, dtype=None):
#         super().__init__()
#         self.num_channels = num_channels
#         self.h_in = h_in
#         self.h_out = h_out
#         self.bias = bias
#         self.depthwise = nn.Conv1d(
#             in_channels=num_channels, out_channels=num_channels*h_out, kernel_size=h_in,
#             stride=1, padding=0, dilation=1,  # default parameters
#             groups=num_channels, bias=bias, device=device, dtype=dtype)
#
#     def _check_input_dim(self, x):
#         if x.ndim < 2:
#             raise ValueError(f"Expected input of shape [*, C, H_in] to have at least 2 dimensions, got {list(x.shape)}")
#         if x.shape[-2] != self.num_channels:
#             raise ValueError(f"Expected {self.num_channels} channels, got {list(x.shape)}")
#         if x.shape[-1] != self.h_in:
#             raise ValueError(f"Expected H_in={self.h_in}, got {list(x.shape)}")
#
#     def forward(self, x) -> torch.Tensor:
#         self._check_input_dim(x)
#         x, ps = pack([x], "* c h_in")   # convert to [n, c, h_in] for Conv1d
#         # x is [n, c, h_in], weight is [c, h_in, h_out], bias is [c, h_out]
#         x = self.depthwise(x)   # n c h_in, c h_in h_out -> n c h_out
#         x = rearrange(x, "n (c h_out) 1 -> n c h_out", c=self.num_channels)
#         [x] = unpack(x, ps, "* c h_out")
#         return x

