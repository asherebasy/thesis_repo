# Dummy dataset
# 0.8 MSELoss + 0.2 L1Loss
# Adam Optimizer with lr=0.0005
# batch_size = 10
# layer_config = [(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
# latent_dim = 128,
# activation_func = 'ReLU',
import torch

from core.data.base_dataset import BasicDataset
from core.data.base_dataloader import BaseDataLoader
from core.loss.loss_functions import MSELoss, L1Loss, CombinedLoss

import torchvision.transforms as transforms

from core.models.autoencoder_upsampling import AutoencoderUpsampling
from core.models.verbose_model import VerboseExecution
from core.train.base_trainer import BaseTrainer


def run_experiment_1(params: dict):
    num_epochs = params['num_epochs']
    base_dir = params['base_dir']
    data_dir = params['data_dir']
    model_kwargs = params['model_kwargs']
    verb_model = params['verbose_model']
    transforms_ = transforms.Compose([
        transforms.Pad((1, 81, 0, 81)),
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Resize((428, 256))
    ])
    dummy_dataset = BasicDataset(
        data_dir=data_dir,
        applied_transforms=transforms_)
    dummy_dataloader = BaseDataLoader(dataset=dummy_dataset,
                                      batch_size=10,
                                      shuffle=True,
                                      validation_split=0.5,
                                      num_workers=4)

    dummy_model = AutoencoderUpsampling(**model_kwargs)
    if verb_model:
        verbose_model = VerboseExecution(dummy_model)
        dummy_input = next(iter(dummy_dataloader))
        _ = verbose_model(dummy_input)

    loss_function_1 = MSELoss()
    loss_function_2 = L1Loss()

    loss_function = CombinedLoss(loss_function_1, loss_function_2, weight1=1, weight2=0)
    opt = torch.optim.Adam(dummy_model.parameters(), lr=0.005)
    device = torch.device("mps")
    # Create an instance of the trainer
    trainer = BaseTrainer(model=dummy_model,
                          base_dir=base_dir,
                          criterion=loss_function,
                          optimizer=opt,
                          dataloader=dummy_dataloader,
                          device=device,
                          save_every=50,
                          save_filters=50,
                          save_feature_maps=50,
                          monitor_cpu_resources_every=20,
                          reconstruct_every=2,
                          plot_loss_every=5,
                          reduce_lr_on_plateu=False
                          )

    # Train the model for a specified number of epochs
    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    model_kwargs = dict(input_channels=3,
                        input_dim=488,
                        layer_config=[
                            (8, 3, 2, 1), (16, 3, 2, 1), (32, 3, 2, 1), #(64, 3, 2, 1), (64, 3, 2, 1)
                        ],
                        latent_dim=514,
                        activation_func_enc='LeakyReLU',
                        activation_func_dec='LeakyReLU',
                        activation_func_mid='LeakyReLU',
                        upsampling_mode='bilinear',
                        batch_norm=True
                        )
    run_experiment_1(params={'num_epochs': 250,
                             'base_dir': '/Users/ammar.elsherebasy/Desktop/Thesis/Training Logs',
                             'data_dir': '/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/one_image',
                             'model_kwargs': model_kwargs,
                             'verbose_model': False
                             })
