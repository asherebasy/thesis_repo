import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders. From https://github.com/victoresque/pytorch-template
    """
    def __init__(self, dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 0,
                 collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


if __name__ == '__main__':
    from core.data.base_dataset import BasicDataset
    import torchvision.transforms as transforms

    transforms = transforms.Compose([
        transforms.Pad((1, 79, 0, 79)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.Resize((428, 256))
    ])
    dummy_dataset = BasicDataset(data_dir='/Users/ammar.elsherebasy/Desktop/Thesis/2023.05.28_validation_of_filtering/subset',
                                 applied_transforms=transforms)
    dataloader = BaseDataLoader(dataset=dummy_dataset,
                                batch_size=10,
                                shuffle=True,
                                validation_split=0.1,
                                num_workers=8
                                )
    what = dataloader.split_validation()
