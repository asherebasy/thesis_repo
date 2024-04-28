import torch

from core.data.base_dataset import BasicDataset
from core.data.base_dataloader import BaseDataLoader
from core.loss.loss_functions import MSELoss, L1Loss, CombinedLoss

import torchvision.transforms as transforms

from core.models.autoencoder import Autoencoder
from core.models.autoencoder_pooling import AutoencoderPooling
from core.models.verbose_model import VerboseExecution, PrintLayer
from core.train.base_trainer import BaseTrainer


def run_experiment_1(params: dict):
    num_epochs = params['num_epochs']
    base_dir = params['base_dir']
    data_dir = params['data_dir']
    model_kwargs = params['model_kwargs']

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad((1, 81, 0, 81)),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((244, 244))
    ])
    dummy_dataset = BasicDataset(
        data_dir=data_dir,
        applied_transforms=transforms_)
    dummy_dataloader = BaseDataLoader(dataset=dummy_dataset,
                                      batch_size=10,
                                      shuffle=True,
                                      validation_split=0.5,
                                      num_workers=4)

    dummy_model = AutoencoderPooling(**model_kwargs)
    verbose_model = VerboseExecution(dummy_model)
    dummy_input = torch.randn((1, 3, model_kwargs['input_dim'], model_kwargs['input_dim']))
    _ = verbose_model(dummy_input)
    print(dummy_model)

    loss_function_1 = MSELoss()
    loss_function_2 = L1Loss()

    loss_function = CombinedLoss(loss_function_1, loss_function_2, weight1=1, weight2=0)
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
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
                          plot_loss_every=5,
                          reconstruct_every=10,
                          reduce_lr_on_plateu=False
                          )

    # Train the model for a specified number of epochs
    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    model_kwargs = dict(input_channels=3,
                        input_dim=244,
                        layer_config=[
                            (16, 3, 1, 1, 2), (16, 3, 1, 1, 2), (32, 3, 2, 1, None)#, (32, 3, 1, 1, 2),
                        ],
                        latent_dim=2000,
                        activation_func='ReLU',
                        batch_norm=True,
                        )
    run_experiment_1(params={'num_epochs': 2,
                             'base_dir': '/Users/ammar.elsherebasy/Desktop/Thesis/Training Logs',
                             'data_dir': '/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/one_image',
                             'model_kwargs': model_kwargs
                             })
