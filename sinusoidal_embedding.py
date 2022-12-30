import math
import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, n=10000):
        """
        A non-trainable sinusoidal embedding layer
        that maps each real-valued scalar to a vector.

        - Input: `(*)`, tensor of arbitrary shape containing the entries to encode.
        - Output: `(*, embedding_dim)`, where `*` is the input shape.

        Sinusoidal embedding for positional encoding is popularized by
        the Transformer architecture to address the order-invariance of attention
        (Attention Is All You Need, https://arxiv.org/abs/1706.03762, by Vaswani et al.).
        Each non-negative-integer index is mapped to a dense real-valued vector;
        this vector embedding is then added with a learned embedding of the same length.

        Output embedding size `embedding_dim` should be an even number `2*half_dim`.
        The output tensor `PE` has 2 components: `PE_sin` and `PE_cos` both have length `half_dim`.
        They are then concatenated to `PE = [PE_sin, PE_cos]` of length `embedding_dim`.
        Taking `n=10000` for example, for any input scalar `x`,
        `PE_sin(x)[i] = sin(x / 10000^{i/(half_dim-1)})`;
        `PE_cos(x)[i] = cos(x / 10000^{i/(half_dim-1)})`.
        Here `i` is zero-indexed, ranging from `0` to `half_dim-1` inclusive.
        So the sinusoidal wavelength ranges from `2pi` (when `i = 0`) to `10000*2pi` (when `i = half_dim-1`) inclusive.
        Note that the `-1` in `half_dim-1` ensures `10000*2pi` wavelength is included.

        This code is adapted from HuggingFace (https://huggingface.co/blog/annotated-diffusion),
        which we use as a benchmark for comparison.
        We generalize this embedding layer, so it applies to
        arbitrary real-valued input tensors (not just [N]-shaped sequences).

        Parameters
        ----------
        embedding_dim : int
            The length of each embedding vector.

        n : float, default: 10000
            Constant to control the maximum wavelength (period), which is `n*2pi`.

            The original transformer paper uses 10000.
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal embedding"
        self.embedding_dim = embedding_dim
        self.n = n

    def forward(self, x: torch.Tensor):  # output shape: x.shape + [dim]
        device = x.device
        half_dim = self.embedding_dim // 2

        # original formula:
        # divisor = self.n ** (torch.arange(half_dim, device=device) / (half_dim - 1))
        # half_embeddings = x[..., None] / divisor

        # HuggingFace implementation:
        # not sure why exp and log are used. Perhaps multiplication is faster?
        # The 2 implementations are mathematically equivalent
        # They differ by ~1e-13 mean MSE but fail the torch.allclose test.
        multiplier = torch.exp(-(math.log(self.n) / (half_dim - 1)) * torch.arange(half_dim, device=device))
        half_embeddings = x[..., None] * multiplier

        embeddings = torch.cat((half_embeddings.sin(), half_embeddings.cos()), dim=-1)
        return embeddings

