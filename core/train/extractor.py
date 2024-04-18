# Importing libraries
import torchvision.models as models
from torch import nn

from utils.utils import nearest_square


# import time


class Extractor:

    def __init__(self, model_children, DS_layer_name='downsample'):
        self.model_children = model_children
        self.DS_layer_name = DS_layer_name

        self.CNN_layers = []
        self.Linear_layers = []
        self.Transpose_layers = []
        self.DS_layers = []

        self.CNN_weights = []
        self.Linear_weights = []
        self.Transpose_weights = []

        self.CNN_layer_names = []
        self.Linear_layer_names = []
        self.Transpose_layer_names = []

        self.__no_sq_layers = 0  # number of sequential layers
        self.__no_containers = 0  # number of containers

        self.__verbose = []

        self.__bottleneck = models.resnet.Bottleneck
        self.__basicblock = models.resnet.BasicBlock

    def __Append(self, layer, layer_type: str):
        """
        This function will append the layers weights and
        the layer itself to the appropriate variables

        params: layer: takes in CNN or Linear layer
        params: layer_type: either 'linear', 'convolutionl', or 'transpose'
        returns: None
        """
        layer_name = str(layer)

        if layer_type == 'linear':
            self.Linear_weights.append(layer.weight)
            self.Linear_layers.append(layer)
            self.Linear_layer_names.append(layer_name)

        elif layer_type == 'convolutional':
            self.CNN_weights.append(layer.weight)
            self.CNN_layers.append(layer)
            self.CNN_layer_names.append(layer_name)

        elif layer_type == 'transpose':
            self.Transpose_weights.append(layer.weight)
            self.Transpose_layers.append(layer)
            self.Transpose_layer_names.append(layer_name)

        else:
            raise ValueError(f'{layer_type} is not a valid type!')

    def __Layer_Extractor(self, layers):
        """
        This function(algorithm) finds CNN and linear layer in a Sequential layer

        params: layers: takes in either CNN or Sequential or linear layer
        return: None
        """
        for x in range(len(layers)):
            if type(layers[x]) in [nn.Sequential, nn.ModuleList]:
                # Calling the fn to loop through the layer to get CNN layer
                self.__Layer_Extractor(layers[x])
                self.__no_sq_layers += 1

            if type(layers[x]) == nn.Conv2d:
                self.__Append(layers[x], 'convolutional')

            if type(layers[x]) == nn.Linear:
                self.__Append(layers[x], 'linear')

            if type(layers[x]) == nn.ConvTranspose2d:
                self.__Append(layers[x], 'transpose')

            # This statement makes sure to get the down-sampling layer in the model
            if self.DS_layer_name in layers[x]._modules.keys():
                self.DS_layers.append(layers[x]._modules[self.DS_layer_name])

            # # The below statement will loop throgh the containers and append it
            # if isinstance(layers[x], (self.__bottleneck, self.__basicblock)):
            #     self.__no_containers += 1
            #     for child in layers[x].children():
            #         if type(child) == nn.Conv2d:
            #             self.__Append(child)

    def __Verbose(self):

        for cnn_l, cnn_wts in zip(self.CNN_layers, self.CNN_weights):
            self.__verbose.append(f"CNN Layer : {cnn_l} ---> Weights shape : {cnn_wts.shape}")

        for linear_l, linear_wts in zip(self.Linear_layers, self.Linear_weights):
            self.__verbose.append(f"Linear Layer : {linear_l}  ---> Weights shape : {linear_wts.shape}")

    def activate(self):
        """Activates the algorithm"""

        # start = time.time()
        self.__Layer_Extractor(self.model_children)
        self.__Verbose()
        # self.__ex_time = str(round(time.time() - start, 5)) + ' sec'

    def info(self):
        """Information"""

        return {
            'Down-sample layers name': self.DS_layer_name,
            'Total CNN Layers': len(self.CNN_layers),
            'Total Sequential Layers': self.__no_sq_layers,
            'Total Downsampling Layers': len(self.DS_layers),
            'Total Linear Layers': len(self.Linear_layers),
            'Total Transpose Layers': len(self.Transpose_layers),
            'Total number of Bottleneck and Basicblock': self.__no_containers,
            # 'Total Execution time': self.__ex_time
        }

    def __repr__(self):
        return '\n'.join(self.__verbose)

    def __str__(self):
        return '\n'.join(self.__verbose)


if __name__ == '__main__':
    from core.models.autoencoder import Autoencoder
    from matplotlib import pyplot as plt

    model = Autoencoder(
        input_channels=3,
        input_dim=428,
        layer_config=[(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
        latent_dim=128,
        activation_func='ReLU'
    )

    extractor = Extractor(model_children=list(model.children()))
    extractor.activate()
