import math
import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, min_period=2*math.pi, max_period=10000*2*math.pi, progression="geometric"):
        """
        A non-trainable element-wise sinusoidal embedding layer
        that maps each real-valued scalar entry to a vector.

        - Input: `(*)`, tensor of arbitrary shape containing the entries to encode.
        - Output: `(*, embedding_dim)`, where `*` is the input shape.

        Output embedding size `embedding_dim` should be an even number `2*half_dim`.
        The output tensor `PE` has 2 components: `PE_sin` and `PE_cos`.
        They both have length `half_dim` in the last (-1) dimension.
        They are then concatenated to `PE = [PE_sin, PE_cos]`
        of length `embedding_dim` in the last (-1) dimension.

        This embedding is performed element-wise.
        For any scalar input `x`, sinusoidal functions `sin(x/n)` and `cos(x/n)`
        have period (wavelength) `2pi*n`.
        (Intuitive naming convention for those familiar with Fourier analysis.)
        The minimum and maximum period lengths are specified as hyperparameters,
        which are then used to derive `n = period/(2*pi)`.
        We then create `half_dim` different `n`s ranging from `min_n` to `max_n`
        according to a geometric or arithmetic progression.
        These `n`s will be used to determine wavelengths in `PE_sin` and `PE_cos`.

        Sinusoidal embedding for positional encoding is popularized by
        the Transformer architecture to address the order-invariance of attention
        (Attention Is All You Need, https://arxiv.org/abs/1706.03762, by Vaswani et al.).
        Each non-negative-integer index is mapped to a dense real-valued vector;
        this vector embedding is then added with a learned embedding of the same length.

        Let's see how the original sinusoidal embedding works in Transformer:
        Taking `min_n=1, max_n=10000, progression="geometric"` for example (as in Transformer),
        for any input scalar `x`,
        `PE_sin(x)[i] = sin(x / 10000^{i/(half_dim-1)})`;
        `PE_cos(x)[i] = cos(x / 10000^{i/(half_dim-1)})`.
        Here `i` is zero-indexed, ranging from `0` to `half_dim-1` inclusive.
        So the sinusoidal wavelength ranges from `2pi` (when `i = 0`) to `10000*2pi` (when `i = half_dim-1`) inclusive.
        Note that the `-1` in `half_dim-1` ensures `10000*2pi` wavelength is included.
        We avoid this `-1` issue altogether by using `torch.linspace`.

        This code is inspired by HuggingFace (https://huggingface.co/blog/annotated-diffusion),
        which we use as a benchmark for comparison.
        We generalize this embedding layer, so it applies to
        arbitrary real-valued input tensors (not just [N]-shaped sequences).
        We also generalize it to arbitrary `min_n`, `max_n`, and `progression` modes
        (Transformer uses geometric progression, we also allow arithmetic progression).

        Parameters
        ----------
        embedding_dim : int
            The length of each embedding vector.

        min_period : float, default: 2*math.pi
            Constant to control the minimum wavelength (period), which is `n*2pi`.

            We choose to use period instead of n, because it saves
            the mental toll to recalculate wavelength given n.

        max_period : float, default: 10000*2*math.pi
            Constant to control the maximum wavelength (period), which is `n*2pi`.

            The original transformer paper uses n=10000.
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal embedding"
        self.embedding_dim = embedding_dim
        assert 0 < min_period < max_period, "min_period must be less than max_period; they both need to be positive"
        self.min_period, self.max_period = min_period, max_period
        self.min_n = min_period / (2 * math.pi)
        self.max_n = max_period / (2 * math.pi)
        assert progression in ("geometric", "arithmetic"), "progression mode must be either geometric or arithmetic"
        self.progression = progression

    def forward(self, x: torch.Tensor):  # output shape: x.shape + [dim]
        device = x.device
        half_dim = self.embedding_dim // 2

        # original formula:
        # divisor = self.max_n ** (torch.arange(half_dim, device=device) / (half_dim - 1))
        # half_embeddings = x[..., None] / divisor

        # HuggingFace implementation:
        # not sure why exp and log are used. Perhaps multiplication is faster?
        # The 2 implementations are mathematically equivalent
        # naive exponentiation & HuggingFace differ by ~1e-13 mean MSE but fail the torch.allclose test.
        # multiplier = torch.exp(-(math.log(self.max_n) / (half_dim - 1)) * torch.arange(half_dim, device=device))
        # half_embeddings = x[..., None] * multiplier

        # My implementation
        # supports min_n, max_n, and geometric/arithmetic progression
        # uses linspace instead of arange; this makes exp log more useful
        if self.progression == "geometric":
            multiplier = torch.exp(torch.linspace(-math.log(self.min_n), -math.log(self.max_n), steps=half_dim, device=device))
        elif self.progression == "arithmetic":
            multiplier = 1 / torch.linspace(self.min_n, self.max_n, steps=half_dim, device=device)
        else:
            raise ValueError(f"Unknown progression mode {self.progression}")
        half_embeddings = x[..., None] * multiplier

        embeddings = torch.cat((half_embeddings.sin(), half_embeddings.cos()), dim=-1)
        return embeddings

