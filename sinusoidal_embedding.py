import math

import numpy as np
import torch
from torch import nn

from typing import Sequence, Tuple


class SinusoidalEncoding(nn.Module):
    def __init__(self,
                 embedding_dim: int = None,
                 period_range: Tuple[float, float] = None,
                 freq_range: Tuple[float, float] = None,
                 progression: str = "geometric",
                 periods: Sequence[float] = None,
                 frequencies: Sequence[float] = None):
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
        So the sinusoidal wavelength ranges from
        `2pi` (when `i = 0`) to `10000*2pi` (when `i = half_dim-1`) inclusive.
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

        Notes
        -----
        Internally, the angular frequency list `self.w` is stored as a numpy float64 array.
        If the input tensor is a float tensor,
        `self.w` will be cast to the input tensor's dtype.
        Otherwise, `self.w` will be cast to torch's default float type (usually float32).

        Parameters
        ----------
        embedding_dim : int, default: None
            The length of each embedding vector.

            If None, `periods` or `frequencies` argument must be provided.
            Then `embedding_dim` will be inferred as `2*len(periods)`.

        period_range : tuple of float, default: None
            The minimum and maximum of linear periods `T`,
            parsed as `(min_period, max_period)`.

            The default values `(2*math.pi, 10000*2*math.pi)` are
            the choice of the original Transformer.

        freq_range : tuple of float, default: None
            The minimum and maximum of linear frequencies `f`.
            parsed as `(min_freq, max_freq)`.

        progression : str, default: "geometric"
            One of "geometric" or "arithmetic" to indicate progression type
            from `min_period` to `max_period` (or from `min_freq` to `max_freq`).

            If you want to customize more complicated progression types,
            consider using `periods`/`frequencies` to pass in a list of periods/frequencies.

        periods : list of float, default: None
            A list of custom linear periods for sinusoidal embedding.

            This argument will override `embedding_dim`, `period_range`, and `progression`.
            For example, `embedding_dim` will become `2*len(periods)` because there's
            one sine and one cosine transformation for each wavelength.

            You cannot pass in `periods` and `frequencies` at the same time.

        frequencies : list of float, default: None
            A list of custom linear frequencies for sinusoidal embedding.

            This argument will override `embedding_dim`, `freq_range`, and `progression`.
            For example, `embedding_dim` will become `2*len(frequencies)`.

            You cannot pass in `periods` and `frequencies` at the same time.

        Attributes
        ----------
        embedding_dim : int
            The embedding dimension

        w : np.ndarray of shape [half_dim]
            The list of angular frequencies used to generate `cos(wx)` and `sin(wx)`.
        """
        super().__init__()

        if periods is not None:         # custom list of linear periods
            assert len(periods) > 0, "list of linear periods must be nonempty"
            self.w = math.tau / np.array([float(x) for x in periods])       # 2pi/T
            return
        if frequencies is not None:     # custom list of linear frequencies
            assert len(frequencies) > 0, "list of linear frequencies must be nonempty"
            self.w = math.tau * np.array([float(x) for x in frequencies])   # 2pi*f
            return

        # use arithmetic/geometric progression with endpoints to generate w
        # some preparation useful for both frequency and period modes of specification
        assert progression in ("geometric", "arithmetic"), "progression mode must be either geometric or arithmetic"
        assert embedding_dim > 0, "embedding_dim must be positive"
        assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal embedding"
        embedding_dim = int(embedding_dim)
        half_dim = embedding_dim // 2

        if freq_range is not None:      # progression from min_freq to max_freq
            min_freq, max_freq = freq_range
            assert 0 < min_freq < max_freq, "min_freq must be less than max_freq; they both need to be positive"
            if progression == "geometric":  # tau * freq
                self.w = np.geomspace(min_freq * math.tau, max_freq * math.tau, num=half_dim, endpoint=True)
            elif progression == "arithmetic":  # tau * freq
                self.w = np.linspace(min_freq * math.tau,  max_freq * math.tau, num=half_dim, endpoint=True)
            else:
                raise ValueError(f"Unknown progression mode {progression}")
            return
        else:                           # progression from min_period to max_period
            if period_range is None:
                period_range = math.tau, 10000 * math.tau  # tau=2pi is better than pi!
            min_period, max_period = period_range
            assert 0 < min_period < max_period, "min_period must be less than max_period; they both need to be positive"
            if progression == "geometric":      # tau / period
                self.w = 1 / np.geomspace(min_period/math.tau, max_period/math.tau, num=half_dim, endpoint=True)
            elif progression == "arithmetic":   # tau / period
                self.w = 1 / np.linspace(min_period/math.tau, max_period/math.tau, num=half_dim, endpoint=True)
            else:
                raise ValueError(f"Unknown progression mode {progression}")
            return

    @property
    def embedding_dim(self):
        return len(self.w) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # output shape: x.shape + [embedding_dim]
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
        w = torch.from_numpy(self.w).to(
            device=x.device,
            dtype=x.dtype if torch.is_floating_point(x) else torch.get_default_dtype())
        half_embeddings = x[..., None] * w

        embeddings = torch.cat((half_embeddings.sin(), half_embeddings.cos()), dim=-1)
        return embeddings

