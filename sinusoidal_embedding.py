import math
import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim=None,
                 min_period=2*math.pi, max_period=10000*2*math.pi, progression="geometric",
                 periods=None):
        """
        A non-trainable element-wise sinusoidal embedding layer
        that maps each real-valued scalar entry to a vector.

        - Input: `(*)`, tensor of arbitrary shape containing the entries to encode.
        - Output: `(*, embedding_dim)`, where `*` is the input shape.

        This embedding is performed element-wise.
        In short, a list of different angular frequencies `w` is generated;
        we compute `sin(wx)` and `cos(wx)` for each `w` and
        concatenate them along the embedding dimension.

        Output embedding size `embedding_dim` should be an even number `2*half_dim`.
        The output tensor `PE` has 2 components: `PE_sin` and `PE_cos`.
        They both have length `half_dim` in the last (-1) dimension.
        They are then concatenated to `PE = [PE_sin, PE_cos]`
        of length `embedding_dim` in the last (-1) dimension.

        For a periodic function `g(x)`, the linear period `T` is defined as
        the length of a complete cycle, such that :math:`g(x)=g(x+T)` for all `x`.
        The linear frequency :math:`f=1/T` counts the number of periods
        in a unit (1) interval of `x`.
        In a sinusoid, the angular frequency :math:`\omega=2\pi f=2\pi/T` is
        defined as the number of radians rotated per unit interval of `x`.
        Angular frequency is helpful because `sin(wx)` and `cos(wx)` have
        angular frequency `w`, linear frequency `f`, and linear period `T`.

        There are 4 ways to generate angular frequencies `w`.
        Only one of the following should be used at runtime:

        1. Use the `periods` keyword argument to pass in a list of linear periods
        2. Generate an arithmetic/geometric progression of `half_dim` linear periods
           from `min_period` to `max_period` inclusive.
        3. Use the `frequencies` keyword argument to pass in a list of linear frequencies
        4. Generate an arithmetic/geometric progression of `half_dim` linear frequencies
           from `min_freq` to `max_freq` inclusive.

        We use linear frequency and period as hyperparameters instead of angular,
        because they let us think in terms of the periodicity in input feature x,
        which rarely relates to :math:`2\pi` in practice.

        Sinusoidal embedding for positional encoding is popularized by
        the Transformer architecture to address the order-invariance of attention
        (Attention Is All You Need, https://arxiv.org/abs/1706.03762, by Vaswani et al.).
        Each non-negative-integer index is mapped to a dense real-valued vector;
        this vector embedding is then added with a learned embedding of the same length.

        Let's see how the original sinusoidal embedding works in Transformer:
        Taking `min_period=2*math.pi, max_period=10000*2*math.pi, progression="geometric"`,
        for any input scalar `x`,
        `PE_sin(x)[i] = sin(x / 10000^{i/(half_dim-1)})`;
        `PE_cos(x)[i] = cos(x / 10000^{i/(half_dim-1)})`.
        Here `i` is zero-indexed, ranging from `0` to `half_dim-1` inclusive.
        So the sinusoidal wavelength ranges from `2pi` (when `i = 0`) to `10000*2pi` (when `i = half_dim-1`) inclusive.
        Note that the `-1` in `half_dim-1` ensures `10000*2pi` wavelength is included.
        We avoid this `-1` issue altogether by using `torch.linspace`.

        This code is inspired by HuggingFace (https://huggingface.co/blog/annotated-diffusion),
        which we use as a benchmark for comparison.

        We generalize this sinusoidal embedding layer in the following ways:

        1. It applies to arbitrary real-valued input tensors, not just [N]-shaped sequences.
        2. It allows arbitrary `min_period` and `max_period` (or `max_freq` and `min_freq`)
           and progression modes (arithmetic/geometric) to generate a sequence of frequencies,
           while Transformer uses geometric progression from `w=1` to `w=10000` angular frequencies.
        3. You can pass in your own list of linear frequencies/periods.

        Parameters
        ----------
        embedding_dim : int, default: None
            The length of each embedding vector.

            If None, `periods` argument must be provided.
            Then `embedding_dim` will be inferred as `2*len(periods)`.

        min_period : float, default: 2*math.pi
            Constant to control the minimum wavelength (period), which is `n*2pi`.

            We choose to use period instead of n, because it saves
            the mental toll to recalculate wavelength given n.

        max_period : float, default: 10000*2*math.pi
            Constant to control the maximum wavelength (period), which is `n*2pi`.

            The original transformer paper uses n=10000.

        progression : str, default: "geometric"
            One of "geometric" or "arithmetic" to indicate progression type
            from min_period to max_period.

        periods : list of float, default: None
            A list of wavelengths (periods) for sinusoidal embedding.

            This argument will override all previous arguments.
            For example, `embedding_dim` will become `2*len(periods)` because there's
            one sine and one cosine transformation for each wavelength.
        """
        super().__init__()
        if periods is None:     # use geometric/arithmetic progression to generate periods
            self.periods = None
            assert embedding_dim > 0, "embedding_dim must be positive"
            assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal embedding"
            self.embedding_dim = int(embedding_dim)
            assert 0 < min_period < max_period, "min_period must be less than max_period; they both need to be positive"
            self.min_period, self.max_period = min_period, max_period
            self.min_n = min_period / (2 * math.pi)
            self.max_n = max_period / (2 * math.pi)
            assert progression in ("geometric", "arithmetic"), "progression mode must be either geometric or arithmetic"
            self.progression = progression
        else:
            self.periods = [float(p) for p in periods]

    def forward(self, x: torch.Tensor):  # output shape: x.shape + [dim]
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
        # multiplier is (2 pi / period), so it can be multiplied with x before taking sine/cosine
        if self.periods is None:
            half_dim = self.embedding_dim // 2
            if self.progression == "geometric":
                multiplier = torch.exp(torch.linspace(-math.log(self.min_n), -math.log(self.max_n), steps=half_dim,
                                                      device=x.device))
            elif self.progression == "arithmetic":
                multiplier = 1 / torch.linspace(self.min_n, self.max_n, steps=half_dim,
                                                device=x.device)
            else:
                raise ValueError(f"Unknown progression mode {self.progression}")
        else:
            multiplier = (2*math.pi) / torch.tensor(self.periods, device=x.device)
            print("periods multiplier dtype", multiplier.dtype)
        half_embeddings = x[..., None] * multiplier

        embeddings = torch.cat((half_embeddings.sin(), half_embeddings.cos()), dim=-1)
        return embeddings

