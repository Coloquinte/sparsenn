import sparsenn
import torch
import pdb


def get_actual_params_linear(layer):
    weight = torch.zeros((layer.out_features, layer.in_features))
    input = torch.zeros((1, layer.in_features))
    output_shape = (1, layer.out_features)
    bias = layer(input).flatten()
    for i in range(layer.in_features):
        input[0, i] = 1.0
        output = layer(input)
        assert output.shape == output_shape
        weight[:, i] = output.flatten()
        input[0, i] = 0.0
    return weight - bias.reshape(-1, 1), bias

def get_actual_params_conv1d(layer):
    weight_shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
    weight = torch.zeros(weight_shape)
    input_shape = (1, layer.in_channels, *layer.kernel_size)
    input = torch.zeros(input_shape)
    output_shape = (1, layer.out_channels, *[1 for i in layer.kernel_size])
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            input[0, i, x] = 1.0
            output = layer(input)
            assert output.shape == output_shape
            weight[:, i, x] = output.flatten()
            input[0, i, x] = 0.0
    return weight - bias.reshape(-1, 1, 1), bias

def get_actual_params_conv2d(layer):
    weight_shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
    weight = torch.zeros(weight_shape)
    input_shape = (1, layer.in_channels, *layer.kernel_size)
    input = torch.zeros(input_shape)
    output_shape = (1, layer.out_channels, *[1 for i in layer.kernel_size])
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            for y in range(layer.kernel_size[1]):
                input[0, i, x, y] = 1.0
                output = layer(input)
                assert output.shape == output_shape
                weight[:, i, x, y] = output.flatten()
                input[0, i, x, y] = 0.0
    return weight - bias.reshape(-1, 1, 1, 1), bias

def get_actual_params_conv3d(layer):
    weight_shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
    weight = torch.zeros(weight_shape)
    input_shape = (1, layer.in_channels, *layer.kernel_size)
    input = torch.zeros(input_shape)
    output_shape = (1, layer.out_channels, *[1 for i in layer.kernel_size])
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            for y in range(layer.kernel_size[1]):
                for z in range(layer.kernel_size[2]):
                    input[0, i, x, y, z] = 1.0
                    output = layer(input)
                    assert output.shape == output_shape
                    weight[:, i, x, y, z] = output.flatten()
                    input[0, i, x, y, z] = 0.0
    return weight - bias.reshape(-1, 1, 1, 1, 1), bias

for hdim in range(1, 5):
    in_size = 5 * 2**hdim
    out_size = 4 * 2**hdim
    l = sparsenn.HypercubeLinear(in_features=in_size, out_features=out_size, hdim=hdim)
    w, b = get_actual_params_linear(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 5):
    in_size = 3 * 2**hdim
    out_size = 2 * 2**hdim
    kernel_size = 3
    l = sparsenn.HypercubeConv1d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, hdim=hdim)
    w, b = get_actual_params_conv1d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 5):
    in_size = 7 * 2**hdim
    out_size = 3 * 2**hdim
    kernel_size = 3
    l = sparsenn.HypercubeConv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, hdim=hdim)
    w, b = get_actual_params_conv2d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 4):
    in_size = 2 * 2**hdim
    out_size = 5 * 2**hdim
    kernel_size = 3
    l = sparsenn.HypercubeConv3d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, hdim=hdim)
    w, b = get_actual_params_conv3d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

in_size = 10
out_size = 7
l = sparsenn.SparseLinear(in_features=in_size, out_features=out_size, sparsity=0.2)
w, b = get_actual_params_linear(l)
assert ((w - l.weight())**2).mean() <= 1.0e-6
assert ((b - l.bias())**2).mean() <= 1.0e-6
l = sparsenn.SparseConv1d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, sparsity=0.2)
w, b = get_actual_params_conv1d(l)
assert ((w - l.weight())**2).mean() <= 1.0e-6
assert ((b - l.bias())**2).mean() <= 1.0e-6
l = sparsenn.SparseConv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, sparsity=0.2)
w, b = get_actual_params_conv2d(l)
assert ((w - l.weight())**2).mean() <= 1.0e-6
assert ((b - l.bias())**2).mean() <= 1.0e-6
l = sparsenn.SparseConv3d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, sparsity=0.2)
w, b = get_actual_params_conv3d(l)
assert ((w - l.weight())**2).mean() <= 1.0e-6
assert ((b - l.bias())**2).mean() <= 1.0e-6