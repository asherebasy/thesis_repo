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
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer_, _, output: print(f"{layer_.__name__}: {output.shape}")
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
