import torch
from torchvision import transforms

from core.data.base_dataloader import BaseDataLoader
from core.data.base_dataset import BasicDataset
from core.models.autoencoder_upsampling import AutoencoderUpsampling
from core.train.visualizer import plot_resconstruction

if __name__ == '__main__':
    state_dict_path = '/Users/ammar.elsherebasy/Desktop/Thesis/Training Logs/checkpoints/AutoencoderUpsampling_20240302_131407/checkpoint_epoch_200.pth'
    dummy_model = AutoencoderUpsampling(
        input_channels=3,
        input_dim=488,
        layer_config=[(8, 3, 2, 1), (16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
        latent_dim=1028,
        activation_func='LeakyReLU',
        upsampling_mode='bilinear',
        state_dict_path=state_dict_path
    )
    transforms_ = transforms.Compose([
        transforms.Pad((1, 81, 0, 81)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Resize((428, 256))
    ])
    validation_dataset = BasicDataset(
        data_dir='/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/one_image', applied_transforms=transforms_
    )
    validation_dataloader = BaseDataLoader(dataset=validation_dataset, batch_size=1, validation_split=0.5)
    vv = validation_dataloader.split_validation()
    validation_data = next(iter(vv))
    reconstructed_data = dummy_model.reconstruct(validation_data)
    plot_resconstruction(test_data=validation_data, reconstructed_data=reconstructed_data)
