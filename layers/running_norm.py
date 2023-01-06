import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = None,
                 device=None, dtype=None) -> None:
        """
        A non-trainable preprocessing module that standardizes features by
        transforming each feature to zero mean and unit variance,
        like StandardScaler in Scikit-Learn, but also supports minibatch training
        and can be integrated into a sequential NN model.

        - Input: `(N, C, *)`, tensor to normalize; `C` is the number of features.
        - Output: `(N, C, *)`, the same as input shape.

        The formula for the transformed output `z` is
        `z = (x - u) / s`, where `u` is  the running mean and
        `s` is the running standard deviation.

        This module differs from BatchNorm in that:

        - Running statistics (mean and variance) are always updated during training only
        - Running statistics (instead of batch statistics) are detached
          and used during both training and evaluation.
        - There are no learnable affine parameters

        For this reason, RunningNorm should only be used as the first layer
        in a neural network.

        Parameters
        ----------
        num_features : int
            Number of features or channels :math:`C` of the input

        eps : float, default: 1e-5
            A value added to the denominator for numerical stability.

        momentum : float, default: None
            The value used for the `running_mean` and `running_var` computation.
            Default to `None` for cumulative moving average (i.e. simple average).

        """
        super(RunningNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.num_batches_tracked: torch.Tensor
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        # running_mean/running_var/num_batches... are registered at runtime
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def _check_input_dim(self, x):
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        if x.shape[1] != self.num_features:
            # my modification, maybe bad for lazy initialization?
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[1]} features")

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"

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

        reduce_axes = [0] + list(range(2, x.ndim))  # reduce over all except feature axis (1)

        batch_mean = x.detach().mean(dim=reduce_axes)
        batch_mean_x2 = (x.detach()**2).mean(dim=reduce_axes)
        batch_var = batch_mean_x2 - batch_mean ** 2
        if self.training:   # update running statistics
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var

        # standard scaling
        stats_shape = [1] * x.ndim
        stats_shape[1] = self.num_features
        mean = self.running_mean.view(*stats_shape)
        std = torch.sqrt(self.running_var + self.eps).view(*stats_shape)
        return (x - mean) / std
