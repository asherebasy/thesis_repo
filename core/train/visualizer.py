import collections
from functools import partial
from typing import DefaultDict, List, Tuple

import torch
import torchvision
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn

from utils.utils import nearest_square


def visualize_layer_features(
        layer_weights: nn.Parameter,
        save_path: str = '',
        title_str: str = ''
) -> None:
    """
    Visualize all the filters of a CNN layer.
    :param layer_weights: the weights of the layer
    :param save_path: path to save the visulization at. Doesn't save if it is an empty string
    :param title_str: optional title of the figure
    """
    nrows, ncols = nearest_square(layer_weights.shape[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(35, 35))
    if nrows == ncols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for index, kernel in enumerate(layer_weights):
        ax = axs[index]
        ax.imshow(kernel[0, :, :].cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
    for remove_ix in range(len(layer_weights), len(axs)):
        axs[remove_ix].axis('off')

    fig.suptitle(t=title_str, x=0.5, y=0, ha='center', va='bottom', fontsize=55)
    if save_path != '':
        fig.savefig(save_path, dpi='figure', bbox_inches='tight', facecolor='white')
    else:
        fig.show()


def save_activations(
        activations: DefaultDict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
) -> None:
    """
    """
    activations[name].append(out.detach().cpu())


def register_activation_hooks(
        model: nn.Module,
        layers_to_save: List[str]
) -> DefaultDict[List, torch.Tensor]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save.

    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = collections.defaultdict(list)

    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations, activations_dict, name)
            )
    return activations_dict


def plot_feature_maps(
        to_save: List,
        model: nn.Module,
        test_data: torch.Tensor,
        aggregate_feature_maps: bool = True,
        save_path: str = '',
) -> None:
    """
    To use: to_save = [mod[0] for mod in list(model.named_modules()) if mod[0]!='' and type(mod[1]) == nn.Conv2d]
            plot_saved_activations(to_save, model, test_data, save_path)
    :param to_save:
    :param model:
    :param test_data:
    :param aggregate_feature_maps:
    :param save_path:
    :return:
    """
    saved_activations = register_activation_hooks(model, layers_to_save=to_save)
    data = test_data
    _ = model(data)

    max_activations = 0  # maximum number of activations; used for visualization purpose
    for activation in saved_activations:
        activation = saved_activations[activation][0]
        max_activations = max(max_activations, activation.squeeze().size()[0])

    for act_number, activation in enumerate(saved_activations.keys()):
        act = saved_activations[activation][0].squeeze()
        counter = act_number * max_activations
        if len(act.shape) > 3:
            random_batch_ix = torch.randint(low=0, high=act.shape[0], size=(1,))[0]
        else:
            act = act.unsqueeze(0)  # add a batch dimension if it is not there
            random_batch_ix = 0
        random_batch = act[random_batch_ix]

        if aggregate_feature_maps:
            # flattened_random_batch = random_batch.flatten(start_dim=-2)
            # kmeans = KMeans(n_clusters=4, random_state=0)
            # clustered_feature_maps = kmeans.fit_predict(flattened_random_batch.numpy())
            # clustered_feature_maps.reshape(0, random_batch.shape[-1], random_batch.shape[-1])
            raise NotImplementedError  # TODO: implement aggregation of feature maps
        else:
            nrows, ncols = nearest_square(random_batch.size(0))
            fig, axs = plt.subplots(nrows, ncols, figsize=(40, 40))  # a figure for every layer
            axs = axs.flatten()
            for idx in range(random_batch.size(0)):  # loop through the feature maps of one layer
                ax_ = axs[idx]
                if data.shape[1] == 3:
                    ax_.imshow(random_batch[idx])
                elif data.shape[1] == 1:
                    ax_.imshow(random_batch[idx], cmap='gray')
                else:
                    raise ValueError
                ax_.axis('off')

            for ax_ix_remove in range(random_batch.size(0), len(axs)):
                axs[ax_ix_remove].axis('off')

            fig.suptitle(t=activation, x=0.5, y=0, va='bottom', ha='center', fontsize=100)

            if save_path != '':
                fig.savefig(save_path + f'{activation}.png', bbox_inches='tight')
            else:
                fig.show()
            plt.close(fig)


def plot_resconstruction(test_data: torch.Tensor,
                         reconstructed_data: torch.Tensor,
                         max_n_samples: int = 1,
                         save_path: str = '') -> (mpl.figure, mpl.axes.Axes):
    original_images = torchvision.utils.make_grid(test_data[:max_n_samples], normalize=True, nrow=1)
    reconstructed_images = torchvision.utils.make_grid(reconstructed_data[:max_n_samples], normalize=True, nrow=1)

    # Display original and reconstructed images
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    ax = axs[0]
    ax.set_title('Original Images')
    ax.imshow(original_images.permute(1, 2, 0))
    ax.axis('off')

    ax = axs[1]
    ax.set_title('Reconstructed Images')
    ax.imshow(reconstructed_images.permute(1, 2, 0))
    ax.axis('off')

    if save_path != '':
        fig.savefig(save_path, dpi='figure', bbox_inches='tight')
    else:
        fig.show()

    return fig, axs


if __name__ == '__main__':
    from core.models.autoencoder import Autoencoder
    from core.train.extractor import Extractor
    import torchvision.transforms as t

    dummy_model = Autoencoder(
        input_channels=3,
        input_dim=428,
        layer_config=[(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
        latent_dim=128,
        activation_func='ReLU'
    )

    extractor = Extractor(model_children=list(dummy_model.children()))
    extractor.activate()
    layer_weights_to_visualize = extractor.CNN_weights[-1]
    # visualize_layer_features(layer_weights_to_visualize)

    dummy_input = torch.randn((4, 3, 428, 428))
    dummy_layers = ['encoder.0', 'encoder.2', 'encoder.4', 'encoder.6',
                    'decoder.0', 'decoder.2', 'decoder.4', 'decoder.6']
    # plot_feature_maps(to_save=dummy_layers, model=dummy_model, test_data=dummy_input, aggregate_feature_maps=False,
    #                   save_path='')

    plot_resconstruction(model=dummy_model, test_data=dummy_input)
