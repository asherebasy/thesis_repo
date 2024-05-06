import os
from pickle import dump, load

import psutil
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.data.base_dataloader import BaseDataLoader
from core.loss.loss_functions import CombinedLoss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from core.train.extractor import Extractor
from core.train.visualizer import visualize_layer_features, plot_feature_maps, plot_resconstruction


class BaseTrainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 dataloader: BaseDataLoader,
                 device: str,
                 base_dir: str = '',
                 save_every: int = 20,
                 save_filters: int = 20,
                 save_feature_maps: int = 20,
                 reconstruct_every: int = 50,
                 plot_loss_every: int = 10,
                 empty_gpu_every: int = 20,
                 monitor_cpu_resources_every: int = None,
                 accum_iter: int = 1,
                 reduce_lr_on_plateu: bool = True
                 ) -> None:
        self.model = model.to(device)
        self.model.apply(self.init_weights)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.validation_dataloader = self.dataloader.split_validation()
        self.device = device
        self.base_dir = base_dir
        self.save_every = save_every
        self.save_filters = save_filters
        self.save_feature_maps = save_feature_maps
        self.reconstruct_every = reconstruct_every
        self.accum_iter = accum_iter
        self.plot_loss_every = plot_loss_every
        self.empty_gpu_every = empty_gpu_every
        self.monitor_cpu_resources_every = monitor_cpu_resources_every
        self.train_loss_dict = {}
        self.reduce_lr_on_plateu = reduce_lr_on_plateu
        if self.reduce_lr_on_plateu:
            self.scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15)
            self.learning_rates = []

        if isinstance(criterion, CombinedLoss):
            comb_loss_weights = str(criterion.weight1) + '_' + str(criterion.weight2)
        else:
            comb_loss_weights = ''

        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = type(self.model).__name__

        loss_name = self.criterion.__class__.__name__
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints",
                                           f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        self.filters_dir = os.path.join(self.base_dir, "filters",
                                        f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        self.feature_maps_dir = os.path.join(self.base_dir, "feature_maps",
                                             f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        self.log_dir = os.path.join(self.base_dir, "logs",
                                    f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        self.train_loss_dir = os.path.join(self.base_dir, "loss",
                                           f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        self.reconstruction_dir = os.path.join(self.base_dir, "reconstruct",
                                               f"{self.model_name}_{self.current_time}_{loss_name}_{comb_loss_weights}")
        for directory in [self.checkpoint_dir, self.filters_dir, self.feature_maps_dir, self.log_dir,
                          self.train_loss_dir, self.reconstruction_dir]:
            os.makedirs(directory, exist_ok=True)

        self.writer = None  # Tensorboard writer
        self.global_step = 0
        self.start_time = None

        self.log_dir = None
        self._initialize_tensorboard_writer()

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def _initialize_tensorboard_writer(self):
        # Set up Tensorboard writer with log directory named after the model and current time
        self.writer = SummaryWriter(self.log_dir)

    def _save_checkpoint(self, epoch, loss):
        # Save model checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    def save_train_loss(self):
        with open(os.path.join(self.train_loss_dir, 'train_loss.pkl'), 'wb') as file:
            dump(self.train_loss_dict, file)

    def plot_train_loss(self):
        train_loss = load(open(os.path.join(self.train_loss_dir, 'train_loss.pkl'), 'rb'))
        train_values = train_loss.values()
        fig, axs = plt.subplots(figsize=(12, 8))
        axs.plot(range(1, self.global_step + 1), train_values, label='Training Loss')
        fig.savefig(os.path.join(self.train_loss_dir, 'train_loss.png'))
        plt.close(fig)

    def monitor_cpu_resources(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        print(
            f'Epoch: {self.global_step}, CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%, '
            f'Disk Usage: {disk_percent}%'
        )

    def reconstruct(self, save_path: str):
        validation_data = next(iter(self.validation_dataloader)).to(self.device)
        reconstructed_data = self.model.reconstruct(validation_data)
        fig, axs = plot_resconstruction(test_data=validation_data.cpu(), reconstructed_data=reconstructed_data.cpu(),
                                        save_path=save_path)
        plt.close(fig)

    def train(self, num_epochs):
        self.model.train()
        self.start_time = datetime.now()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0

            inputs = None
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
                self.optimizer.zero_grad()
                inputs = batch
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            self.global_step += 1
            average_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch}/{num_epochs}, Average Loss: {average_loss:.4f}")
            self.train_loss_dict[epoch] = average_loss
            if self.reduce_lr_on_plateu:
                self.scheduler.step(average_loss)
                self.learning_rates.append(self.scheduler.optimizer.param_groups[0]['lr'])
            # Log loss to Tensorboard
            self.writer.add_scalar('Loss', average_loss, self.global_step)

            if epoch % self.save_every == 0:
                # Save model checkpoint at the end of each epoch
                self._save_checkpoint(epoch, average_loss)

            if epoch % self.save_filters == 0:
                extractor = Extractor(model_children=list(self.model.children()))
                extractor.activate()
                epoch_filter_dir = os.path.join(self.filters_dir, f'epoch_{epoch}')
                os.makedirs(epoch_filter_dir, exist_ok=True)
                for name, layer in zip(extractor.CNN_layer_names, extractor.CNN_weights):
                    layer_features_save_path = os.path.join(epoch_filter_dir, name) + '.png'
                    visualize_layer_features(layer_weights=layer,
                                             save_path=layer_features_save_path,
                                             title_str=name)

            if epoch % self.save_feature_maps == 0:
                feature_map_layers = [f'encoder.{i}' for i in range(0, 14)]
                feature_map_layers += [f'decoder.{i}' for i in range(0, 14)]
                layer_feature_maps_save_path = os.path.join(self.feature_maps_dir, f'epoch_{epoch}')
                os.makedirs(layer_feature_maps_save_path, exist_ok=True)
                plot_feature_maps(to_save=feature_map_layers,
                                  model=self.model,
                                  test_data=inputs,
                                  aggregate_feature_maps=False,
                                  save_path=layer_feature_maps_save_path + '/')

            if epoch % self.plot_loss_every == 0:
                self.save_train_loss()
                self.plot_train_loss()

            if epoch % self.empty_gpu_every and self.device == 'mpu':
                torch.mps.empty_cache()
            elif epoch % self.empty_gpu_every and self.device == 'gpu':
                torch.cuda.empty_cache()
            else:
                pass

            if self.monitor_cpu_resources_every:
                if epoch % self.monitor_cpu_resources_every:
                    self.monitor_cpu_resources()

            if epoch % self.reconstruct_every == 0 and self.validation_dataloader is not None:
                reconstruct_save_path = os.path.join(self.reconstruction_dir, f'epoch_{epoch}.png')
                self.reconstruct(save_path=reconstruct_save_path)

        self.writer.close()
        print("Training completed.")


if __name__ == "__main__":
    from core.data.base_dataset import BasicDataset
    from core.data.base_dataloader import BaseDataLoader
    from core.models.autoencoder import Autoencoder
    from core.loss.loss_functions import MSELoss, L1Loss, CombinedLoss

    import torchvision.transforms as transforms

    transforms = transforms.Compose([
        transforms.Pad((1, 79, 0, 79)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Resize((428, 256))
    ])
    dummy_dataset = BasicDataset(
        data_dir='/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/subset',
        applied_transforms=transforms)
    dummy_dataloader = BaseDataLoader(dataset=dummy_dataset,
                                      batch_size=10,
                                      shuffle=True,
                                      validation_split=0.05,
                                      num_workers=4)

    dummy_model = Autoencoder(
        input_channels=3,
        input_dim=428,
        layer_config=[(16, 3, 2, 1), (32, 3, 2, 1), (64, 3, 2, 1), (64, 3, 2, 1)],
        latent_dim=128,
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
                          base_dir='/Users/ammar.elsherebasy/Desktop/Thesis/Training Logs',
                          criterion=loss_function,
                          optimizer=opt,
                          dataloader=dummy_dataloader,
                          device=device,
                          save_every=50,
                          save_filters=50,
                          save_feature_maps=50,
                          plot_loss_every=2,
                          monitor_cpu_resources_every=2
                          )

    # Train the model for a specified number of epochs
    trainer.train(num_epochs=10)
