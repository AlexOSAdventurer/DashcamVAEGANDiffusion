import torch
from data import DiffSet, ImageDataset
import pytorch_lightning as pl
from model import AutoencoderKL, DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml

# Training hyperparameters
dataset_choice = "AutoencoderBoosted256x256"
base_dir = "/work/cseos2g/papapalpi/DeepDriveStuff/bdd100k/images/"
dataset_path_train = base_dir + "data/train_float_256x256.npy"
dataset_path_val = base_dir + "data/val_float_256x256.npy"
latent_dataset_path_train =  base_dir + "data/train_float_256x256_latent.npy"
latent_dataset_path_val =  base_dir + "data/val_float_256x256_latent.npy"
max_epoch = 10
batch_size = 16
config_data = yaml.safe_load(open("autoencoder_kl_256x256x3.yaml"))

# Loading parameters
load_model = False
load_version_num = 34

# Code for optionally loading model
pass_version = None
last_checkpoint = None

if load_model:
    pass_version = load_version_num
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]


# Create datasets and data loaders
train_dataset = ImageDataset(dataset_path_train, latent_dataset_path_train)
val_dataset = ImageDataset(dataset_path_val, latent_dataset_path_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

autoencoder_model = AutoencoderKL.load_from_checkpoint("autoencoderkl.ckpt", ddconfig=config_data['model']['params']['ddconfig'],
                 lossconfig=config_data['model']['params']['lossconfig'],
                 embed_dim=config_data['model']['params']['embed_dim'],
                 base_learning_rate=config_data['model']['base_learning_rate'])

model = DiffusionModel(autoencoder_model=autoencoder_model, in_size=64*64, in_size_sqrt=64, t_range=1000, img_depth=train_dataset.depth, train_dataset=train_dataset)

# Load Trainer model
tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name=dataset_choice,
    version=pass_version,
)

def getModel() -> pl.LightningModule:
    return model

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        workers: int = 2,
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
# TODO: determine per-process batch size given total batch size
# TODO: enable evaluate
cli.trainer.fit(cli.model, datamodule=cli.datamodule)
