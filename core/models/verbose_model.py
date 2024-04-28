from torch import nn, Tensor


class VerboseExecution(nn.Module):
    """
    Class to print the output shape of the input as it goes through the layers of a model.
    Use as: verbose_model = VerboseExecution(model().to(device))
            _ = verbose_model(input)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for idx, layer in enumerate(self.model.modules()):
            if isinstance(layer, nn.Sequential):
                # If the layer is Sequential, we iterate over its children
                for i, sub_layer in enumerate(layer.children()):
                    sub_layer.__name__ = f"{type(sub_layer).__name__}_{idx}_{i}"
                    sub_layer.register_forward_hook(
                        lambda module, input, output, name=sub_layer.__name__: print(f"{name}: {output.shape}")
                    )
            else:
                # For other types of layers
                layer.__name__ = f"{type(layer).__name__}_{idx}"
                layer.register_forward_hook(
                    lambda module, input, output, name=layer.__name__: print(f"{name}: {output.shape}")
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
