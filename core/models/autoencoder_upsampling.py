from typing import Tuple

import numpy as np
import torch
from torch import nn

from core.models.base_model import BaseModel
from utils.utils import calculate_conv_output_size


class AutoencoderUpsampling(BaseModel):
    def __init__(self,
                 layer_config: list,
                 latent_dim: int,
                 input_dim: int,
                 input_channels: int = 3,
                 batch_norm: bool = False,
                 activation_func_enc: str = None,
                 activation_func_dec: str = None,
                 activation_func_mid: str = None,
                 state_dict_path: str = None,
                 upsampling_mode: str = 'nearest'
                 ) -> None:
        super().__init__()
        self.encoder, self.decoder, self.middle_block = None, None, None
        self.encoder_layer_sizes = [input_dim]
        self.middle_block_to_decoder_size = None
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.layer_config = layer_config
        self.input_channels = input_channels
        self.batch_norm = batch_norm
        self.act_func_encoder = self._build_activation_function(activation_func_enc)
        self.act_func_decoder = self._build_activation_function(activation_func_dec)
        self.activation_func_middle = self._build_activation_function(activation_func_mid)
        self.upsampling_mode = upsampling_mode

        # build encoder and decoder
        _ = self._build_encoder()
        self._build_middle_block()
        self._build_decoder()

        self.epoch = 0
        self.state_dict_path = state_dict_path
        if state_dict_path:
            self.load_model_from_state_dict(path=state_dict_path)
            epoch = state_dict_path.split('_epoch_')[1]
            epoch = int(epoch.split('.')[0])
            self.epoch = epoch

    def _build_encoder(self) -> int:
        self.encoder = nn.ModuleList()
        if not isinstance(self.layer_config, list):
            raise RuntimeError(f"Expected layer configuration to be a list, but got {type(self.layer_config)}")

        input_channels = self.input_channels
        for layer in self.layer_config:
            # check the validity of current tuple entry
            if not isinstance(layer, Tuple):
                raise ValueError(f"Expected a tuple, but got {type(layer)}")
            if len(layer) != 4:
                raise RuntimeError(f"Expected a tuple of 4 entries, but got {len(layer)}")

            output_size = calculate_conv_output_size(input_size=self.encoder_layer_sizes[-1],
                                                     kernel_size=layer[1],
                                                     padding=layer[3],
                                                     stride=layer[2])
            self.encoder_layer_sizes.append(output_size)
            self.encoder.extend(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=layer[0],
                        kernel_size=layer[1],
                        stride=layer[2],
                        padding=layer[3]
                    ),
                    nn.BatchNorm2d(layer[0]) if self.batch_norm else nn.Identity(),
                    self.act_func_encoder
                )
            )
            input_channels = layer[0]

        return input_channels

    def _build_decoder(self) -> None:
        self.decoder = nn.ModuleList()
        if not isinstance(self.layer_config, list):
            raise RuntimeError(f"Expected layer configuration to be a list, but got {type(self.layer_config)}")

        for ix, layer in enumerate(reversed(self.layer_config)):
            # check the validity of current tuple entry
            if not isinstance(layer, Tuple):
                raise ValueError(f"Expected a tuple, but got {type(layer)}")
            if len(layer) != 4:
                raise RuntimeError(f"Expected a tuple of 4 entries, but got {len(layer)}")

            input_channels = layer[0]
            try:
                output_channels = self.layer_config[-ix-2][0]
            except IndexError:
                output_channels = self.input_channels

            # output_padding = calculate_output_padding(input_size=self.encoder_layer_sizes[-ix - 1],
            #                                           output_size=self.encoder_layer_sizes[-ix - 2],
            #                                           stride=layer[2],
            #                                           kernel_size=layer[1],
            #                                           padding=layer[3])
            self.decoder.extend(
                nn.Sequential(
                    nn.Upsample(
                        size=self.encoder_layer_sizes[-ix-2],
                        mode=self.upsampling_mode,
                        align_corners=True if self.upsampling_mode != 'nearest' else None
                    ),
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=layer[1],
                        stride=1,
                        padding=layer[3],
                    ),
                    nn.BatchNorm2d(output_channels) if self.batch_norm else nn.Identity(),
                    self.act_func_decoder
                )
            )
            self.decoder.extend(nn.Sequential(nn.Tanh()))

    def _build_middle_block(self) -> None:
        with torch.no_grad():
            middle_input_size = self._calculate_features()

        self.middle_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(middle_input_size, self.latent_dim * 2),
            self.activation_func_middle,
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            self.activation_func_middle,
            nn.Linear(self.latent_dim, self.latent_dim * 2, ),
            self.activation_func_middle,
            nn.Linear(self.latent_dim * 2, middle_input_size),
            self.activation_func_middle
        )

    def _calculate_features(self) -> int:
        # inputs are always square
        _tensor = torch.rand(size=(1, self.input_channels, self.input_dim, self.input_dim))
        _tensor = self.encode(_tensor)
        self.middle_block_to_decoder_size = tuple(_tensor.shape[1:])
        fts_size = int(np.prod(_tensor.size()[1:]))
        del _tensor
        return fts_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.middle_block(x)
        x = x.view((x.size(0),) + self.middle_block_to_decoder_size)
        x = self.decode(x)

        return x


# if __name__ == '__main__':
    # model = AutoencoderUpsampling(
    #     input_channels=3,
    #     input_dim=428,
    #     layer_config=[(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
    #     latent_dim=128,
    #     activation_func='LeakyReLU',
    #     upsampling_mode='bilinear'
    # )
    # dummy_input = torch.randn((1, 3, 428, 428))
    # verbose_model = VerboseExecution(model)
    # _ = verbose_model(dummy_input)
    #
    # reconstructions = model.reconstruct(dummy_input)
