import torch
from torch import nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma=1., trainable: bool = False,
                 device=None, dtype=None):
        """
        For input feature vector `x`, apply linear transformation
        followed by concatenated sine and cosine: `[sin(Ax), cos(Ax)]`.

        - Input: `(*, in_features)`, where the last dimension is the feature dimension
        - Output: `(*, out_features)`, where all but the last dimension are the same shape as the input

        Unlike element-wise sinusoidal encoding,
        Fourier features capture the interaction of high-dimensional input features.

        Note that a bias term is not necessary in the linear transformation,
        because `sin(wx+b)` and `cos(wx+b)` can both be represented by
        linear combinations of `sin(wx)` and `cos(wx)`, which can be learned in the
        subsequent linear layer.
        (https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

        In learnable Fourier features (https://arxiv.org/abs/2106.02795),
        the linear transformation is trainable after random normal initialization.
        They also use an MLP (linear, GELU, linear) following this Fourier feature layer,
        because the goal is to produce a positional encoding added to word embedding.
        We do not implement the MLP here, since a user should be free to decide
        which layers to follow; this keeps the modularity of FourierFeatures.

        Also note that in learnable Fourier features, the final output
        after sine/cosine is divided by sqrt(H_out). We do not implement that yet.

        Parameters
        ----------
        in_features: int
            Size of each input sample

        out_features: int
            Size of each output sample

        sigma: float
            Standard deviation of 0-centered normal distribution,
            which is used to initialize linear transformation parameters.

        trainable: bool, default: False
            If True, the linear transformation is trainable by gradient descent.
        """
        super(FourierFeatures, self).__init__()
        assert out_features % 2 == 0, "number of out_features must be even"
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma  # standard deviation for normal initialization
        self.trainable = bool(trainable)

        # define linear layer
        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features // 2,
            bias=False,
            device=device, dtype=dtype)

        # initialize weights
        nn.init.normal_(self.linear.weight.data, mean=0.0, std=self.sigma)

        # freeze layer if not trainable
        for param in self.linear.parameters():
            param.requires_grad_(self.trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x


