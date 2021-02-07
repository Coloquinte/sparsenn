import sparsenn
import torch
import pdb

def get_actual_params_conv1d(layer):
    weight = torch.zeros((layer.out_channels, layer.in_channels, layer.kernel_size[0]))
    input = torch.zeros((1, layer.in_channels, layer.kernel_size[0]))
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            input[0, i, x] = 1.0
            weight[:, i, x] = layer(input).flatten()
            input[0, i, x] = 0.0
    return weight - bias.reshape(-1, 1, 1), bias

for hdim in range(1, 7):
    l = sparsenn.HypercubeLinear(in_features=2**hdim, out_features=2**hdim, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim

for hdim in range(1, 7):
    kernel_size = 1
    l = sparsenn.HypercubeConv1d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size
    l.debug = False
    w, b = get_actual_params_conv1d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 7):
    kernel_size = 3
    l = sparsenn.HypercubeConv2d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size**2

for hdim in range(1, 5):
    kernel_size = 3
    l = sparsenn.HypercubeConv3d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size**3