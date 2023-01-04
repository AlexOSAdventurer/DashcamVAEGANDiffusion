import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import sys
import torchgan
import numpy as np
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from contextlib import contextmanager
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, 4, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SAWrapper(nn.Module):
    def __init__(self, h_size, num_s):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.num_s = num_s
        self.h_size = h_size

    def forward(self, x):
        x = x.view(-1, self.h_size, self.num_s * self.num_s).swapaxes(1, 2)
        x = self.sa(x)
        x = x.swapaxes(2, 1).view(-1, self.h_size, self.num_s, self.num_s)
        return x


# U-Net code adapted from: https://github.com/milesial/Pytorch-UNet


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 base_learning_rate
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.learning_rate = base_learning_rate
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
    def encode_raw(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        if (batch_idx == 0):
            self.val_batch = batch
        return self.log_dict
        
    def on_validation_epoch_end(self):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        sample_batch_size = self.val_batch.shape[0]
        decoding, _ = self(self.val_batch)
        x = torch.cat([self.val_batch, decoding], dim=0)
        x = (x.clamp(-1, 1) + 1) / 2.0
        
        tb_logger.add_images(f"val/output_images", x, self.current_epoch)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

class DiffusionModel(pl.LightningModule):
    def __init__(self, autoencoder_model, in_size, in_size_sqrt, t_range, img_depth, train_dataset):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        self.in_size_sqrt = in_size_sqrt
        self.img_depth = img_depth
        self.autoencoder_model = autoencoder_model.eval()
        self.train_dataset = train_dataset
        print("Compiling latents...")
        sys.stdout.flush()
        self.latents_mode = DiagonalGaussianDistribution(torch.from_numpy(train_dataset.latent_data.copy()).type(torch.FloatTensor)).mode()
    
        print(img_depth)
        print("Depth printed again!")
        sys.stdout.flush()
        bilinear = True
        self.inc = DoubleConv(img_depth, 128)
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 512 // factor)
        self.down4 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 256 // factor, bilinear)
        self.up4 = Up(256, 256 // factor, bilinear)
        self.up5 = Up(256, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(128, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 128, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x5 = self.down4(x4) + self.pos_encoding(t, 256, 2)
        #print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
        x = self.up2(x5, x4) + self.pos_encoding(t, 128, 4)
        x = self.up3(x, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up4(x, x2) + self.pos_encoding(t, 128, 16)
        #print(x.shape, x1.shape)
        x = self.up5(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape, device=self.device)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        images, moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        loss = self.get_loss(encoding, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, moments = batch
        posterior = DiagonalGaussianDistribution(moments)
        encoding = posterior.sample()
        loss = self.get_loss(encoding, batch_idx)
        self.log("val/loss", loss)
        if (batch_idx == 0):
            self.val_batch = images
            self.val_encoding = posterior.mode()
        return
    
    def get_nearest_latents(self, x):
        self.latents_mode = self.latents_mode.to(self.device)
        resulting_list = []
        for i in range(x.shape[0]):
            latent = torch.unsqueeze(x[i], 0)
            difference = torch.sum(torch.abs(self.latents_mode - latent), dim=(1,2,3))
            values, indices = torch.topk(difference, 1, largest=False)
            resulting_list.append(int(indices[0]))
        print(resulting_list)
        return self.latents_mode[resulting_list]
    
    def on_validation_epoch_end(self):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        sample_batch_size = 16
        sample_steps = torch.arange(self.t_range-1, 0, -1, device=self.device)
        print("Random seed made!")
        x = torch.randn((sample_batch_size, self.img_depth, self.in_size_sqrt, self.in_size_sqrt), device=self.device)
        for t in sample_steps:
            x = self.denoise_sample(x, t)
        nearest_neighbors = self.get_nearest_latents(x)
        x = self.autoencoder_model.decode(x)
        x = (x.clamp(-1, 1) + 1) / 2.0
        nearest_neighbors = self.autoencoder_model.decode(nearest_neighbors)
        nearest_neighbors = (nearest_neighbors.clamp(-1, 1) + 1) / 2.0
        print("Clamped!")
        print(x.shape)
        tb_logger.add_images(f"val/output_images", x, self.current_epoch)
        tb_logger.add_images(f"val/output_images_nearest", nearest_neighbors, self.current_epoch)
        tb_logger.add_images(f"val/decoded_original_images", (self.autoencoder_model.decode(self.val_encoding).clamp(-1, 1) + 1) / 2.0, self.current_epoch)
        tb_logger.add_images(f"val/original_images", (self.val_batch.clamp(-1, 1) + 1) / 2.0, self.current_epoch)
        print("Images added!")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.inc.parameters())
                                    + list(self.down1.parameters())
                                    + list(self.down2.parameters())
                                    + list(self.down3.parameters())
                                    + list(self.down4.parameters())
                                    + list(self.up2.parameters())
                                    + list(self.up3.parameters())
                                    + list(self.up4.parameters())
                                    + list(self.up5.parameters())
                                    + list(self.sa1.parameters())
                                    + list(self.sa2.parameters())
                                    + list(self.sa3.parameters())
                                    + list(self.outc.parameters()), lr=5e-4)
        return optimizer
