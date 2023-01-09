import torch
from torch import nn

from einops import rearrange


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, h_in, h_out, bias=True,
                 device=None, dtype=None):
        """
        Use n distinct linear layers to separately transform n features
        (each of embedding length `h_in`) to output of length `h_out`.

        - Input: `(N, C, H_in)` or `(N, C)`, the latter is equivalent to `(N, C, 1)`.
        - Output: `(N, C, H_out)`, after each feature embedding vector is processed.

        This is an implementation of feature-tokenizer layer in
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

        Parameters
        ----------
        num_features : int
            The number of features `C` in input tensor of shape `[N, C, H]`.

        h_in : int
            The embedding size of the input features. 1 if there's no embedding.

        h_out : int
            The output embedding size after the layer.

        bias : bool, default: True
            Whether the linear layer has bias.
        """
        super(FeatureTokenizer, self).__init__()
        self.num_features = num_features
        self.h_in = h_in
        self.h_out = h_out
        self.bias = bias
        self.depthwise = nn.Conv1d(
            in_channels=num_features, out_channels=num_features*h_out, kernel_size=h_in,
            stride=1, padding=0, dilation=1,  # default parameters
            groups=num_features, bias=bias, device=device, dtype=dtype)

    def _check_input_dim(self, x):
        if x.ndim not in (2, 3):
            raise ValueError(f"Expected input of shape [N, C] or [N, C, H_in], got {x.shape}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[1]} features")

    def forward(self, x) -> torch.Tensor:
        self._check_input_dim(x)
        if x.ndim == 2:
            x = rearrange(x, "n c -> n c 1")
        x = self.depthwise(x)
        x = rearrange(x, "n (c h) 1 -> n c h", c=self.num_features)
        return x




