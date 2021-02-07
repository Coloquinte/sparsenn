import sparsenn
import torch
import pdb


def get_actual_params_linear(layer):
    weight = torch.zeros((layer.out_features, layer.in_features))
    input = torch.zeros((1, layer.in_features))
    bias = layer(input).flatten()
    for i in range(layer.in_features):
        input[0, i] = 1.0
        output = layer(input)
        weight[:, i] = layer(input).flatten()
        input[0, i] = 0.0
    return weight - bias.reshape(-1, 1), bias

def get_actual_params_conv1d(layer):
    weight_shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
    weight = torch.zeros(weight_shape)
    input_shape = (1, layer.in_channels, *layer.kernel_size)
    input = torch.zeros(input_shape)
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            input[0, i, x] = 1.0
            output = layer(input)
            weight[:, i, x] = output.flatten()
            input[0, i, x] = 0.0
    return weight - bias.reshape(-1, 1, 1), bias

def get_actual_params_conv2d(layer):
    weight_shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
    weight = torch.zeros(weight_shape)
    input_shape = (1, layer.in_channels, *layer.kernel_size)
    input = torch.zeros(input_shape)
    bias = layer(input).flatten()
    output_shape = (1, layer.in_channels, *[1 for i in layer.kernel_size])
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
    bias = layer(input).flatten()
    for i in range(layer.in_channels):
        for x in range(layer.kernel_size[0]):
            for y in range(layer.kernel_size[1]):
                for z in range(layer.kernel_size[2]):
                    input[0, i, x, y, z] = 1.0
                    output = layer(input)
                    weight[:, i, x, y, z] = output.flatten()
                    input[0, i, x, y, z] = 0.0
    return weight - bias.reshape(-1, 1, 1, 1, 1), bias

for hdim in range(1, 7):
    l = sparsenn.HypercubeLinear(in_features=2**hdim, out_features=2**hdim, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim
    #l.debug = False
    w, b = get_actual_params_linear(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 7):
    kernel_size = 3
    l = sparsenn.HypercubeConv1d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size
    l.debug = False
    w, b = get_actual_params_conv1d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 7):
    kernel_size = 1
    l = sparsenn.HypercubeConv2d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size**2
    l.debug = False
    w, b = get_actual_params_conv2d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6

for hdim in range(1, 5):
    kernel_size = 1
    l = sparsenn.HypercubeConv3d(in_channels=2**hdim, out_channels=2**hdim, kernel_size=kernel_size, hdim=hdim)
    assert (l.weight() != 0.0).sum().item() == (hdim+1) * 2**hdim * kernel_size**3
    l.debug = False
    w, b = get_actual_params_conv3d(l)
    assert ((w - l.weight())**2).mean() <= 1.0e-6
    assert ((b - l.bias())**2).mean() <= 1.0e-6