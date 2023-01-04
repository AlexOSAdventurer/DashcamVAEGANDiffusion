import torch
from data import DiffSet, ImageDataset
import pytorch_lightning as pl
from model import AutoencoderKL
from torch.utils.data import DataLoader
import imageio
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy

# Training hyperparameters
dataset_choice = "Testing"
base_dir = "/work/cseos2g/papapalpi/DeepDriveStuff/bdd100k/images/"
dataset_path_train = base_dir + "data/train_float_128x128.npy"
dataset_path_val = base_dir + "data/val_float_128x128.npy"
new_dataset_path_train =  base_dir + "data/train_float_128x128_latent.npy"
new_dataset_path_val =  base_dir + "data/val_float_128x128_latent.npy"
max_epoch = 10
batch_size = 128
config_data = yaml.safe_load(open("autoencoder_kl_64x64x3.yaml"))
device = 'cuda'
# Create datasets and data loaders
train_dataset = ImageDataset(dataset_path_train)
val_dataset = ImageDataset(dataset_path_val)

autoencoder_model = AutoencoderKL.load_from_checkpoint("autoencoderkl.ckpt", ddconfig=config_data['model']['params']['ddconfig'],
                 lossconfig=config_data['model']['params']['lossconfig'],
                 embed_dim=config_data['model']['params']['embed_dim'],
                 base_learning_rate=config_data['model']['base_learning_rate'])

autoencoder_model = autoencoder_model.eval().to(device)

def convertData(dataset, new_path):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=numpy.float, shape=(len(dataset), 6, 32, 32), mode='w+')
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