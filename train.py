import torch
from data import ImageDataset
import pytorch_lightning as pl
from lightning_training_model import DiffusionModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml

# Training hyperparameters
dataset_choice = "AutoencoderBoosted256x256"
base_dir = "/work/cseos2g/papapalpi/"
latent_dataset_path_train =  base_dir + "data/train_float_128x128.npy"
latent_dataset_path_val =  base_dir + "data/val_float_128x128.npy"
config_data = yaml.safe_load(open("diffusion_model_64x64x3.yaml"))

# Loading parameters
load_model = False
load_version_num = 21

# Code for optionally loading model
last_checkpoint = None

if load_model:
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

# Create datasets and data loaders
train_dataset = ImageDataset(latent_dataset_path_train)
val_dataset = ImageDataset(latent_dataset_path_val)

train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, num_workers=16, shuffle=False)

if load_model:
    model = DiffusionModel.load_from_checkpoint(last_checkpoint, config=config_data)
else:
    model = DiffusionModel(config_data)

# Load Trainer model
tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name=dataset_choice,
    version=None,
)

def getModel() -> pl.LightningModule:
    return model

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__()

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return val_loader

cli = LightningCLI(
        description="PyTorch Diffusiom Model with Autoencoder Boost",
        model_class=getModel,
        datamodule_class=ImageDataModule,
        seed_everything_default=123,
        save_config_overwrite=True,
        trainer_defaults=dict(
            accelerator="gpu",
            max_epochs=1000,
            strategy="ddp",
            logger=tb_logger
        ),
)

cli.trainer.fit(cli.model, datamodule=cli.datamodule)
