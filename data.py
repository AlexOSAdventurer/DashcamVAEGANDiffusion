import torch
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, images_path, latent_path=None):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        return torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences