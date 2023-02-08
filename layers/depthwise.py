import math
import torch
from torch import nn

from einops import rearrange, pack, unpack, einsum


class Depthwise1d(nn.Module):
    def __init__(self, num_channels, h_in, h_out, bias=True,
                 device=None, dtype=None):
        """
        Use n distinct linear layers to separately transform n channels,
        such that each channel changes from length `h_in` to length `h_out`.

        - Input: `(*, C, H_in)`, where `∗` means any number of dimensions including none.
        - Output: `(*, C, H_out)`, where all but the last dimension have the same shape
          as the input.

        This is a generalization of feature-tokenizer layer in
        FT-Transformer (https://arxiv.org/abs/2106.11959),
        which uses different linear layers to embed each numerical feature
        to a vector, mapping from `(N, C)` to `(N, C, H_out)`.
        We generalize it to allow input features that are already embedding vectors,
        mapping from `(N, C, H_in)` to `(N, C, H_out)`.
        In the original paper, feature tokenizer is only applied to numerical features,
        but we make this layer modular, so it's left for users to decide
        which architecture to choose (e.g. apply after categorical `nn.Embedding`).

        Under the hood, it is implemented with a `nn.Conv1d` depthwise convolution,
        which is first introduced by MobileNet (https://arxiv.org/abs/1704.04861).
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
        self.num_channels = num_channels
        self.h_in = h_in
        self.h_out = h_out
        self.bias = bias
        self.depthwise = nn.Conv1d(
            in_channels=num_channels, out_channels=num_channels*h_out, kernel_size=h_in,
            stride=1, padding=0, dilation=1,  # default parameters
            groups=num_channels, bias=bias, device=device, dtype=dtype)

    def _check_input_dim(self, x):
        if x.ndim < 2:
            raise ValueError(f"Expected input of shape [*, C, H_in] to have at least 2 dimensions, got {x.shape}")
        if x.shape[-2] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape}")
        if x.shape[-1] != self.h_in:
            raise ValueError(f"Expected H_in={self.h_in}, got {x.shape}")

    def forward(self, x) -> torch.Tensor:
        self._check_input_dim(x)
        x, ps = pack([x], "* c h_in")   # convert to [n, c, h_in] for Conv1d
        # x is [n, c, h_in], weight is [c, h_in, h_out], bias is [c, h_out]
        x = self.depthwise(x)   # n c h_in, c h_in h_out -> n c h_out
        x = rearrange(x, "n (c h_out) 1 -> n c h_out", c=self.num_channels)
        [x] = unpack(x, ps, "* c h_out")
        return x


class CustomDepthwise1d(nn.Module):
    def __init__(self,
                 num_channels: int,
                 h_in: int,
                 h_out: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        # my Depthwise 1d using einsum implementation, presumably simpler
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.h_in = h_in
        self.h_out = h_out
        self.weight = nn.Parameter(torch.empty((num_channels, h_in, h_out), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_channels, h_out), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # we mimic the initialization of nn.Linear:
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        # however, calculation of fan_in and fan_out is not supported for depthwise MLP weights
        bound = 1 / math.sqrt(self.h_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            if self.h_in == 0:
                bound = 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.weight, "... c h_in, c h_in h_out -> ... c h_out")
        if self.bias is not None:
            # x has shape [..., c, h_out], bias has shape [c, h_out]
            x = x + self.bias
        return x


layer = CustomDepthwise1d(num_channels=10, h_in=5, h_out=7, bias=True)
a = torch.rand(3, 10, 5)
print(layer(a).shape)
