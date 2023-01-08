import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, images_path, latent_path=None):
        self.images_data = np.load(images_path, mmap_mode='r')
        if latent_path is not None:
            self.latent_data = np.load(latent_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        if hasattr(self, 'latent_data'):
            return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0), torch.from_numpy(self.latent_data[index].copy()).type(torch.FloatTensor)
        else:
            return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences

class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST"):
        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )

        self.dataset_len = len(train_dataset.data)

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data)
            data = data.unsqueeze(3)
            self.depth = 1
            self.size = 32
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data)
            self.depth = 3
            self.size = 32
        self.input_seq = ((data / 255.0) * 2.0) - 1.0
        self.input_seq = self.input_seq.moveaxis(3, 1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item]
