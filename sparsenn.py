import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _SparseLayer(nn.Module):
    def __init__(self):
        super(_SparseLayer, self).__init__()
        self.optimized = False
        
    def weight(self):
        raise NotImplementedError()
        
    def bias(self):
        raise NotImplementedError()
        
    def forward_optimized(self, x):
        raise NotImplementedError()
    
    def forward_unoptimized(self, x):
        raise NotImplementedError()
        
    def forward(self, x):
        if self.optimized:
            return self.forward_optimized(x)
        else:
            return self.forward_unoptimized(x)


class _SparseLinear(_SparseLayer):
    def __init__(self, in_features, out_features):
        super(_SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
    
    def forward_unoptimized(self, x):
        return F.linear(x, self.weight(), self.bias())

        
class _SparseConv1d(_SparseLayer):
    def __init__(self, in_channels, out_channels, stride, padding, dilation):
        super(_SparseConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def forward_unoptimized(self, x):
        return torch.nn.functional.conv1d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)


class _SparseConv2d(_SparseLayer):
    def __init__(self, in_channels, out_channels, stride, padding, dilation):
        super(_SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def forward_unoptimized(self, x):
        return torch.nn.functional.conv2d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)

        
class _SparseConv3d(_SparseLayer):
    def __init__(self, in_channels, out_channels, stride, padding, dilation):
        super(_SparseConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def forward_unoptimized(self, x):
        return torch.nn.functional.conv3d(
            x, self.weight(), self.bias(),
            self.stride, self.padding, self.dilation)


class HypercubeLinear(_SparseLinear):
    def __init__(self, in_features, out_features, hdim, bias=True):
        super(HypercubeLinear, self).__init__(in_features, out_features)
        assert hdim >= 2
        assert in_features % 2**hdim == 0
        assert out_features % 2**hdim == 0
        self.hdim = hdim
        self._hweight = nn.Parameter(torch.Tensor(
            hdim+1, 2**hdim, out_features//2**hdim, in_features//2**hdim))
        if bias:
            self._hbias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('_hbias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self._hweight, a=math.sqrt(5))
        if self._hbias is not None:
            bound = 1 / math.sqrt(self.in_features * (self.hdim+1))
            nn.init.uniform_(self._hbias, -bound, bound)
        
    def weight(self):
        hdim = self.hdim
        in_features = self.in_features
        out_features = self.out_features
        w = torch.zeros(2**hdim, 2**hdim, out_features//2**hdim, in_features//2**hdim)
        # Make diagonal
        for i in range(2**hdim):
            w[i, i] = self._hweight[0, i]
        # Make dimensions
        for i in range(2**hdim):
            for d in range(hdim):
                j = i ^ (1 << d)  # Neighbour on the hypercube
                w[j, i] = self._hweight[d+1, i]
        w = w.permute(0, 2, 1, 3)
        w = w.reshape(out_features, in_features)
        return w
        
    def bias(self):
        return self._hbias

    
class HypercubeConv1D(_SparseConv1d):
    pass

    
class HypercubeConv2D(_SparseConv2d):
    pass

    
class HypercubeConv3D(_SparseConv3d):
    pass

    
class KroneckerLinear(_SparseLinear):
    def __init__(
        self,
        in_features,
        out_features,
        kronecker_shapes,
        bias=True
        ):
        super(KroneckerLinear, self).__init__(
            in_features, out_features)
        self.kronecker_shapes = kronecker_shapes
        self.factors = []
        for i, s in enumerate(self.kronecker_shapes):
            p = nn.Parameter(torch.Tensor(s))
            self.factors.append(p)
            self.register_parameter(f"factor{i+1}", p)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('_bias', None)
    
    def weight(self):
        w = torch.ones( (1, 1) )
        for t in self.factors:
            w = torch.kron(w, t)
            
    def bias(self):
        return self._bias

     
class KroneckerConv1D(_SparseConv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kronecker_shapes,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        super(KroneckerConv1D, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation)
        self.kronecker_shapes = kronecker_shapes
        self.factors = []
        for i, s in enumerate(self.kronecker_shapes):
            p = nn.Parameter(torch.Tensor(s))
            self.factors.append(p)
            self.register_parameter(f"factor{i+1}", p)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('_bias', None)
    
    def weight(self):
        w = torch.ones( (1, 1, 1) )
        for t in self.factors:
            w = torch.kron(w, t)
            
    def bias(self):
        return self._bias
        
    
class KroneckerConv2D(_SparseConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kronecker_shapes,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        super(KroneckerConv2D, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation)
        self.kronecker_shapes = kronecker_shapes
        self.factors = []
        for i, s in enumerate(self.kronecker_shapes):
            p = nn.Parameter(torch.Tensor(s))
            self.factors.append(p)
            self.register_parameter(f"factor{i+1}", p)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('_bias', None)
    
    def weight(self):
        w = torch.ones( (1, 1, 1, 1) )
        for t in self.factors:
            w = torch.kron(w, t)
            
    def bias(self):
        return self._bias

    
class KroneckerConv3D(_SparseConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kronecker_shapes,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
        ):
        super(KroneckerConv3D, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation)
        self.kronecker_shapes = kronecker_shapes
        self.factors = []
        for i, s in enumerate(self.kronecker_shapes):
            p = nn.Parameter(torch.Tensor(s))
            self.factors.append(p)
            self.register_parameter(f"factor{i+1}", p)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('_bias', None)
    
    def weight(self):
        w = torch.ones( (1, 1, 1, 1, 1) )
        for t in self.factors:
            w = torch.kron(w, t)
            
    def bias(self):
        return self._bias