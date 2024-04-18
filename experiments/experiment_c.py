# Dummy dataset
# 0.8 MSELoss + 0.2 L1Loss
# Adam Optimizer with lr=0.0005
# batch_size = 10
# layer_config = [(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
# latent_dim = 1028
# activation_func = 'ReLU',
import sys

import torch

from core.data.base_dataset import BasicDataset
from core.data.base_dataloader import BaseDataLoader
from core.models.autoencoder import Autoencoder
from core.loss.loss_functions import MSELoss, L1Loss, CombinedLoss

import torchvision.transforms as transforms

from core.train.base_trainer import BaseTrainer


def run_experiment_1(params: dict):
    num_epochs = params['num_epochs']

    transforms_ = transforms.Compose([
        transforms.Pad((1, 79, 0, 79)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Resize((428, 256))
    ])
    dummy_dataset = BasicDataset(
        data_dir='/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/subset',
        applied_transforms=transforms_)
    dummy_dataloader = BaseDataLoader(dataset=dummy_dataset,
                                      batch_size=10,
                                      shuffle=True,
                                      validation_split=0.,
                                      num_workers=4)

    dummy_model = Autoencoder(
        input_channels=3,
        input_dim=428,
        layer_config=[(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
        latent_dim=1028,
        activation_func='ReLU',
        state_dict_path=None
    )

    loss_function_1 = MSELoss()
    loss_function_2 = L1Loss()

    loss_function = CombinedLoss(loss_function_1, loss_function_2, weight1=0.8, weight2=0.2)
    opt = torch.optim.Adam(dummy_model.parameters(), lr=0.0005)
    device = torch.device("mps")
    # Create an instance of the trainer
    trainer = BaseTrainer(model=dummy_model,
                          criterion=loss_function,
                          optimizer=opt,
                          dataloader=dummy_dataloader,
                          device=device,
                          save_every=50,
                          save_filters=50,
                          save_feature_maps=50,
                          )

    # Train the model for a specified number of epochs
    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    run_experiment_1(params={'num_epochs': 200})
    sys.exit()
