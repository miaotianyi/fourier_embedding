from typing import Sequence, Union
from collections import abc

import torch
from torch import nn


class RunningNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(self,
                 kept_axes: Union[int, Sequence[int]],
                 kept_shape: Union[int, Sequence[int]],
                 eps: float = 1e-5,
                 momentum: float = None,
                 device=None, dtype=None) -> None:
        """
        A non-trainable preprocessing module that standardizes input tensor by
        transforming each feature to zero mean and unit variance,
        like StandardScaler in Scikit-Learn, but also supports minibatch training,
        arbitrary input tensor shape and dimensions,
        and can be integrated into a sequential NN model with one line of code.

        - Input: `(*)`, tensor to normalize.
        - Output: `(*)`, the same as input shape.

        The formula for the transformed output `z` is
        `z = (x - u) / s`, where `u` is  the running mean and
        `s` is the running standard deviation.

        The shape of `u` and `s` are both `kept_shape` (which is a list of int).
        They are broadcast to the same dimensions as `x` during runtime.
        Each element in `kept_axes` is regarded as a single feature,
        while the remaining `x` axes are reduced to calculate mean and variance.
        If the data has shape `[N, C]` and the `feature_axes` are `[C]`,
        then there will be `[C]`-shaped mean vector and variance vector,
        and they will be computed across the batch axis `N`.
        It's therefore possible scale by batch, layer, or instance.
        We don't support GroupNorm-like grouping yet.

        This module differs from BatchNorm in that:

        - Running statistics (mean and variance) are always updated during training only
        - Running statistics (instead of batch statistics) are detached
          and used during both training and evaluation.
        - There are no learnable affine parameters

        For this reason, RunningNorm should only be used as the first layer
        in a neural network.

        Parameters
        ----------
        kept_shape : int, sequence of int
            Number of features or channels in each feature axis.

            If there are multiple axes to keep, a list of integers must be provided.
            The running mean and variance will have the same shape as `kept_shape`.

        kept_axes : int, sequence of int, default: (1,)
            The axes to keep; other axes are reduced for mean and variance.

            Must have the same length as `kept_shape`.

        eps : float, default: 1e-5
            A value added to the denominator for numerical stability.

        momentum : float, default: None
            The value used for the `running_mean` and `running_var` computation.
            Default to `None` for cumulative moving average (i.e. simple average).

        """
        super(RunningNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # check kept_axes
        if isinstance(kept_axes, abc.Sequence):  # list of int
            self.kept_axes = tuple(int(s) for s in kept_axes)
        else:   # int
            self.kept_axes = (int(kept_axes), )
        # check kept_shape
        if isinstance(kept_shape, abc.Sequence):
            assert len(kept_shape) == self.ndim,\
                f"kept_shape {kept_shape} and kept_axes {kept_axes} must have the same length"
            self.kept_shape = tuple(int(s) for s in kept_shape)
        else:   # int; broadcast to list if there's more than 1 kept axis
            self.kept_shape = (int(kept_shape),) * self.ndim

        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(*self.kept_shape, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(*self.kept_shape, **factory_kwargs))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.reset_running_stats()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in the running statistics tensors
        """
        return len(self.kept_axes)

    def reset_running_stats(self) -> None:
        # running_mean/running_var/num_batches... are registered at runtime
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def _check_input_dim(self, x):
        if not all(x.shape[a] == s for a, s in zip(self.kept_axes, self.kept_shape)):
            raise ValueError(f"expected shape {self.kept_shape} at axes {self.kept_axes}, got input shape {x.shape}")
        # if x.dim() < 2:
        #     raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        # if x.shape[1] != self.num_features:
        #     # my modification, maybe bad for lazy initialization?
        #     raise ValueError(f"Expected {self.num_features} features, got {x.shape[1]} features")

    def extra_repr(self):
        return f"{self.kept_shape}, kept_axes={self.kept_axes}, eps={self.eps}, momentum={self.momentum}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.

        if self.training:
            self.num_batches_tracked.add_(1)  # type: ignore[has-type]
            if self.momentum is None:  # use cumulative moving average
                # alpha: exponential_average_factor, learning rate, "same" as momentum
                alpha = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                alpha = self.momentum
        else:
            alpha = 0.0

        # represent kept_axes as non-negative integers (so negative kept_axes) can be supported
        srs_axes = tuple(range(x.ndim)[a] for a in self.kept_axes)  # source axes
        tgt_axes = tuple(range(self.ndim))  # target axes
        reduce_axes = tuple(a for a in range(x.ndim) if a not in srs_axes)
        # reduce_axes = [0] + list(range(2, x.ndim))  # reduce over all except feature axis (1)

        batch_mean = x.detach().mean(dim=reduce_axes, keepdim=True).movedim(srs_axes, tgt_axes).squeeze()
        batch_mean_x2 = (x.detach()**2).mean(dim=reduce_axes, keepdim=True).movedim(srs_axes, tgt_axes).squeeze()
        batch_var = batch_mean_x2 - batch_mean ** 2
        if self.training:   # update running statistics
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var

        # running stats back to x
        # stats_shape = [1] * x.ndim
        # stats_shape[1] = self.num_features
        new_idx = (...,) + (None,) * (x.ndim - self.ndim)   # insert trivial columns at the end
        mean = self.running_mean[new_idx].movedim(tgt_axes, srs_axes)
        std = torch.sqrt(self.running_var + self.eps)[new_idx].movedim(tgt_axes, srs_axes)
        # mean = self.running_mean.view(*stats_shape)
        # std = torch.sqrt(self.running_var + self.eps).view(*stats_shape)
        return (x - mean) / std
