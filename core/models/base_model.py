import torch
from torch import nn, Tensor
from abc import abstractmethod


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    @abstractmethod
    def _build_encoder(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _build_decoder(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_middle_block(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _calculate_features(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @staticmethod
    def _build_activation_function(func: str) -> nn.Module:
        if hasattr(nn, func):
            function = getattr(nn, func)

            if isinstance(function, type) and issubclass(function, nn.Module):
                try:
                    return function(inplace=True)
                except TypeError:
                    return function()
            else:
                raise ValueError(f"{function} is not a valid PyTorch class")
        else:
            raise ValueError(f"{func} is not part of nn.Module")

    def load_model_from_state_dict(self,
                                   path: str,
                                   strict: bool = True
                                   ) -> None:
        model_state_dict = torch.load(path)['model_state_dict']
        self.load_state_dict(model_state_dict, strict=strict)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reconstruct(self, test_data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed_data = self.forward(test_data)

        return reconstructed_data
