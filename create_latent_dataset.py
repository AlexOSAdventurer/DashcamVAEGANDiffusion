import torch
from data import OriginalImageDataset as ImageDataset
import pytorch_lightning as pl
import first_stage_autoencoder
from torch.utils.data import DataLoader
import imageio
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy

# Training hyperparameters
dataset_choice = "Testing"
base_dir = "/work/cseos2g/papapalpi/DeepDriveStuff/bdd100k/images/"
dataset_path_train = base_dir + "data/train_float_256x256.npy"
dataset_path_val = base_dir + "data/val_float_256x256.npy"
new_dataset_path_train =  base_dir + "data/train_float_256x256_latent_2.npy"
new_dataset_path_val =  base_dir + "data/val_float_256x256_latent_2.npy"
max_epoch = 10
batch_size = 32
device = 'cuda'
# Create datasets and data loaders
train_dataset = ImageDataset(dataset_path_train)
val_dataset = ImageDataset(dataset_path_val)

autoencoder_model = first_stage_autoencoder.generate_pretrained_model()  
autoencoder_model = autoencoder_model.eval().to(device)

def convertData(dataset, new_path):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=numpy.float, shape=(len(dataset), 6, 64, 64), mode='w+')
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            data = data.to(device)
            result = autoencoder_model.encode_raw(data).to('cpu')
            output_memmap[(i * batch_size):((i * batch_size) + data.shape[0])] = result.numpy()
            print(i)
        
print("Train")
convertData(train_dataset, new_dataset_path_train)
print("Val")
convertData(val_dataset, new_dataset_path_val)
