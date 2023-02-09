import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import sys
import numpy as np
import torch.nn.functional as F
import first_stage_autoencoder
from first_stage_autoencoder.distribution import DiagonalGaussianDistribution
import unet_autoencoder
import custom_diffusion as diffusion

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
        self.unet_autoencoder = unet_autoencoder.generate_model()
        #self.ddim_model = unet_autoencoder.generate_ddim_model()
        #self.semantic_encoder = unet_autoencoder.generate_semantic_encoder_model()
        #Not used during training, but useful for visualization during validation stage
        self.autoencoder_model = first_stage_autoencoder.generate_pretrained_model().eval()
        
    def decode_encoding(self, encoding):
        return (self.autoencoder_model.decode(encoding).clamp(-1, 1) + 1.0) / 2.0
        #return (encoding.clamp(-1, 1) + 1.0) / 2.0
        
    def forward(self, x, t, c=None):
        if c is None:
            c = diffusion.encode_semantic(self.unet_autoencoder.encoder, x)
        return diffusion.estimate_noise(self.unet_autoencoder, x, t, c)
        
    def get_loss(self, images, batch_idx):
        number_of_images = images.shape[0]
        time_steps = diffusion.create_random_time_steps(number_of_images, self.t_range, self.device)
        noised_images, source_noise = diffusion.diffuse_images(images, time_steps, self.t_range, self.beta_small, self.beta_large)
        z_sem = diffusion.encode_semantic(self.unet_autoencoder.encoder, images)
        estimated_noise = self.forward(noised_images, time_steps, z_sem)
        return F.mse_loss(estimated_noise, source_noise)

    def training_step(self, batch, batch_idx):
        moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        #encoding = batch
        loss = self.get_loss(encoding, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        #encoding = batch
        loss = self.get_loss(encoding, batch_idx)
        self.log("val/loss", loss)
        if ((batch_idx == 0) and (self.global_rank == 0)):
            self.val_batch = posterior.mode()
            #self.val_batch = encoding
        return
    
    def on_validation_epoch_end(self):
        if ((self.global_rank != 0) or ((self.current_epoch % 5) != 0)):
            return
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        z_sem = diffusion.encode_semantic(self.unet_autoencoder.encoder, self.val_batch)
        print("Semantic encoding done!")
        x_t = diffusion.stochastic_encode_process_multiple_images(self.unet_autoencoder, self.val_batch, z_sem, self.t_range, self.beta_small, self.beta_large)
        print("Stochastic encoding done!")
        reconstructed_x_0 = diffusion.denoise_process_multiple_images(self.unet_autoencoder, x_t, z_sem, self.t_range, self.beta_small, self.beta_large)
        print("Denoising done!")
        ground_truth_images = self.decode_encoding(self.val_batch)
        reconstructed_images = self.decode_encoding(reconstructed_x_0)
        reconstructed_x_t_images = self.decode_encoding(x_t)
        print("Decoding done!")
        tb_logger.add_images(f"val/original_encoding", self.val_batch, self.current_epoch)
        tb_logger.add_images(f"val/original_images", ground_truth_images, self.current_epoch)
        tb_logger.add_images(f"val/output_images", reconstructed_images, self.current_epoch)
        tb_logger.add_images(f"val/output_encoding", reconstructed_x_0, self.current_epoch)
        tb_logger.add_images(f"val/stochastic_encoding", x_t, self.current_epoch)
        tb_logger.add_images(f"val/stochastic_images", reconstructed_x_t_images, self.current_epoch)
        print("Images added!")

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(list(self.ddim_model.parameters())+list(self.semantic_encoder.parameters()), lr=self.config["model"]["base_learning_rate"])
        optimizer = torch.optim.Adam(list(self.unet_autoencoder.parameters()), lr=self.config["model"]["base_learning_rate"])
        return optimizer
