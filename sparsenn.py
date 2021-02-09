import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _SparseLayer(nn.Module):
    def __init__(self):
        super(_SparseLayer, self).__init__()
        self.debug = False

    def weight(self):
        raise NotImplementedError()

    def bias(self):
        raise NotImplementedError()

    def forward_optimized(self, x):
        raise NotImplementedError()

    def forward_debug(self, x):
        raise NotImplementedError()

    def dense_size(self):
        raise NotImplementedError()

    def sparse_size(self):
        raise NotImplementedError()

    def sparsity(self):
        return self.sparse_size() / self.dense_size()

    def forward(self, x):
        if self.debug:
            return self.forward_debug(x)
        else:
            return self.forward_optimized(x)


class _HypercubeLayer(_SparseLayer):
    def __init__(self, hdim, shape, bias):
        super(_HypercubeLayer, self).__init__()
        assert hdim >= 1
        assert shape[0] % 2**hdim == 0
        assert shape[1] % 2**hdim == 0
        self.hdim = hdim
        self._shape = shape
        wshape = ((hdim+1)*shape[0], shape[1]//2**hdim) + shape[2:]
        self._hweight = nn.Parameter(torch.Tensor(*wshape))
        if bias:
            self._hbias = nn.Parameter(torch.Tensor(shape[0]))
        else:
            self.register_parameter('_hbias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Matches the default Pytorch initialization for dense layers
        # They are written with init.kaiming_uniform_, which obfuscates the true purpose a bit
        fan_in = self._shape[1] * (self.hdim+1) // 2**self.hdim
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self._hweight, -bound, bound)
        if self._hbias is not None:
            nn.init.uniform_(self._hbias, -bound, bound)

    def dense_size(self):
        s = 1
        for i in self._shape:
            s *= i
        return s

    def sparse_size(self):
        return self._hweight.numel()

    def weight(self):
        hdim = self.hdim
        s = self._shape
        hweight = self._hweight.reshape(2**hdim, hdim+1, s[0]//2**hdim, s[1]//2**hdim, *s[2:])
        out = torch.zeros(2**hdim, 2**hdim, s[0]//2**hdim, s[1]//2**hdim, *s[2:])
        # Make diagonal
        for ind in range(2**hdim):
            out[ind, ind] = hweight[ind, 0]
        # Make dimensions
        for ind in range(2**hdim):
            for d in range(hdim):
                out_ind = ind ^ (1 << d)  # Neighbour on the hypercube
                out[out_ind, ind] = hweight[ind, d+1]
        # Merge the 2**hdim dimensions with in/out dims
        out = out.permute(0, 2, 1, 3, *range(4, 2+len(s)))
        out = out.reshape(s)
        return out

    def bias(self):
        return self._hbias

    def merge_hypercube(self, x):
        hdim = self.hdim
        s = self._shape
        unfolded_shape = (x.shape[0], 2**hdim, hdim+1, s[0]//2**hdim, *x.shape[2:])
        output_shape = (x.shape[0], s[0], *x.shape[2:])
        
        x = x.reshape(unfolded_shape)
        out = x.clone()
        for d in range(hdim):
            mid_shape = unfolded_shape[0:1] + (2**(hdim-d-1), 2, 2**d) + unfolded_shape[3:]
            t = x[:, :, d+1].reshape(mid_shape)
            t = torch.roll(t, 1, dims=2)
            out[:, :, d+1] = t.reshape(unfolded_shape[0:2] + unfolded_shape[3:])
        out = out.sum(axis=2).reshape(output_shape)
        return out


class HypercubeLinear(_HypercubeLayer):
    def __init__(self, in_features, out_features, hdim, bias=True):
        super(HypercubeLinear, self).__init__(hdim, (out_features, in_features, 1), bias)
        self.in_features = in_features
        self.out_features = out_features

    def weight(self):
        return super().weight().reshape(self.out_features, self.in_features)

    def forward_debug(self, x):
        return F.linear(x, self.weight(), self.bias())

    def forward_optimized(self, x):
        input_shape = x.shape
        x = torch.nn.functional.conv1d(torch.unsqueeze(x, -1), self._hweight, None,
            groups=2**self.hdim)
        x = torch.squeeze(self.merge_hypercube(x), -1)
        if self._hbias is not None:
            x = x + self._hbias
        return x


class HypercubeConv1d(_HypercubeLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        hdim,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        super(HypercubeConv1d, self).__init__(hdim, (out_channels, in_channels, kernel_size), bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward_debug(self, x):
        return torch.nn.functional.conv1d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)

    def forward_optimized(self, x):
        # Grouped convolution to expand the number of channels
        x = torch.nn.functional.conv1d(x, self._hweight, None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=2**self.hdim)
        # Reorganize the channels to where they belong and sum
        x = self.merge_hypercube(x)
        if self._hbias is not None:
            x = x + self._hbias.reshape(-1, 1)
        return x


class HypercubeConv2d(_HypercubeLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        hdim,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        super(HypercubeConv2d, self).__init__(hdim, (out_channels, in_channels, kernel_size[0], kernel_size[1]), bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward_debug(self, x):
        return torch.nn.functional.conv2d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)

    def forward_optimized(self, x):
        # Grouped convolution to expand the number of channels
        x = torch.nn.functional.conv2d(x, self._hweight, None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=2**self.hdim)
        # Reorganize the channels to where they belong and sum
        x = self.merge_hypercube(x)
        if self._hbias is not None:
            x = x + self._hbias.reshape(-1, 1, 1)
        return x


class HypercubeConv3d(_HypercubeLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        hdim,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        super(HypercubeConv3d, self).__init__(hdim, (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]), bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward_debug(self, x):
        return torch.nn.functional.conv3d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)

    def forward_optimized(self, x):
        # Grouped convolution to expand the number of channels
        x = torch.nn.functional.conv3d(x, self._hweight, None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=2**self.hdim)
        # Reorganize the channels to where they belong and sum
        x = self.merge_hypercube(x)
        if self._hbias is not None:
            x = x + self._hbias.reshape(-1, 1, 1, 1)
        return x
