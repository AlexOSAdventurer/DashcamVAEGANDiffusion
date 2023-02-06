import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import sys
import numpy as np
import torch.nn.functional as F
import first_stage_autoencoder
from first_stage_autoencoder.distribution import DiagonalGaussianDistribution
import diffusion


class DiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.beta_small = config["model"]["beta_small"]
        self.beta_large = config["model"]["beta_large"]
        self.t_range = config["model"]["t_range"]
        self.in_size = config["model"]["in_size"]
        self.in_size_sqrt = config["model"]["in_size_sqrt"]
        self.img_depth = config["model"]["img_depth"]
        self.config = config
        #Not used during training, but useful for visualization during validation stage
        self.autoencoder_model = first_stage_autoencoder.generate_pretrained_model().eval()
        self.ddim_model = None
        self.semantic_encoder = None
        
    def forward(self, x, t, c=None):
        if c is None:
            c = diffusion.encode_semantic(self.semantic_encoder, x)
        return diffusion.estimate_noise(self.ddim_model, x, t, c)
        
    def get_loss(images, batch_idx):
        number_of_images = images.shape[0]
        time_steps = diffusion.create_random_time_steps(number_of_images, self.t_range, self.device)
        noised_images, source_noise = diffusion.diffuse_images(images, time_steps, self.t_range, self.beta_small, self.beta_large)
        estimated_noise = self.forward(noised_images, time_steps)
        return F.mse_loss(estimated_noise, source_noise)

    def training_step(self, batch, batch_idx):
        moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        loss = self.get_loss(encoding, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        loss = self.get_loss(encoding, batch_idx)
        self.log("val/loss", loss)
        if ((batch_idx == 0) and (self.global_rank == 0)):
            self.val_batch = images
            self.val_encoding = posterior.mode()
        return
    
    def on_validation_epoch_end(self):
        if (self.global_rank != 0):
            return
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

        #tb_logger.add_images(f"val/output_images", x, )
        print("Images added!")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.config["model"]["base_learning_rate"])
        return optimizer