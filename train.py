import torch
from data import DiffSet, ImageDataset
import pytorch_lightning as pl
from model import AutoencoderModel
from torch.utils.data import DataLoader
import imageio
import glob
from pytorch_lightning.utilities.cli import LightningCLI

# Training hyperparameters
dataset_choice = "BerkeleyDeepDrive"
dataset_path_train = "/work/cseos2g/papapalpi/data/train_float_128x128.npy"
dataset_path_val = "/work/cseos2g/papapalpi/data/val_float_128x128.npy"
max_epoch = 10
batch_size = 16

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
train_dataset = ImageDataset(dataset_path_train)
val_dataset = ImageDataset(dataset_path_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

# Create model and trainer
if load_model:
    model = AutoencoderModel.load_from_checkpoint(last_checkpoint, in_size=train_dataset.size*train_dataset.size, in_size_sqrt=train_dataset.size, img_depth=train_dataset.depth)
else:
    print(train_dataset.size, train_dataset.depth)
    print("Depth printed!")
    model = AutoencoderModel(train_dataset.size*train_dataset.size, train_dataset.size, train_dataset.depth)

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
        description="PyTorch VAE-GAN Model Training",
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
