import math
from typing import Tuple

import numpy as np
import matplotlib as mpl
import torch
import torchvision
from matplotlib import pyplot as plt


def matplotlib_imshow(
        ax: mpl.axes.Axes,
        img,
        one_channel: bool = False
):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        ax.imshow(npimg)
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis('off')


def visualize_img_grid(
        images: torch.Tensor
):
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img=img_grid, one_channel=False)


def nearest_square(num) -> Tuple:
    """
    Given a num, calculates the nearest (greater than or equal) square root number and returns the two squares of that
    number.

    :param num:
    :return:
    """
    sqrt = math.sqrt(num)
    if sqrt.is_integer():
        return int(sqrt), int(sqrt)
    else:
        side = int(sqrt) + 1
        return side, side


def calculate_conv_output_size(input_size, kernel_size, padding, stride) -> int:
    """
    Calculate the size of the convolutional layer output.

    Parameters:
    - input_size (int): Size of the input feature map.
    - kernel_size (int): Size of the convolutional kernel.
    - padding (int): Amount of zero-padding applied to the input.
    - stride (int): Stride of the convolution operation.

    Returns:
    - int: Size of the convolutional layer output.
    """
    output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
    return output_size


def calculate_output_padding(output_size, input_size, stride, kernel_size, padding) -> int:
    """
    Calculate the output_padding parameter for a transposed convolution.

    Parameters:
    - output_size (int): Target size of the transposed convolution output.
    - input_size (int): Size of the input feature map.
    - stride (int): Stride of the transposed convolution operation.
    - kernel_size (int): Size of the transposed convolutional kernel.
    - padding (int): Amount of zero-padding applied to the input.

    Returns:
    - int: Output_padding parameter for the transposed convolution.
    """
    output_padding = output_size - (1 + kernel_size - 1 + (input_size - 1)*stride - 2*padding)

    return output_padding
