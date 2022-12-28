import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import torchgan
import numpy as np
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

blocks = [2, 2, 2, 2]
block_outdim = [64, 128, 256, 3]
zDim=block_outdim[-1]

def residualBlockDownFullDisc(channelIn, channelOut, blocks=3, downsample=2):
    layers = []
    for i in range(blocks):
        filters = [channelIn, channelOut, channelOut] if (i == 0) else [channelOut, channelOut, channelOut]
        kernels = [3, 3]
        strides = [downsample, 1] if (i == 0) else [1, 1]
        paddings = [1, 1]
        shortcut = nn.Conv2d(channelIn, channelOut, kernel_size=1, stride=downsample, bias=False) if (i == 0) else nn.Identity()
        layer = torchgan.layers.ResidualBlock2d(filters=filters, kernels=kernels, strides=strides, paddings=paddings, 
                                           nonlinearity=nn.LeakyReLU(0.2), batchnorm=False, 
                                           shortcut=shortcut, 
                                           last_nonlinearity=nn.LeakyReLU(0.2))
        layers = layers + [layer]
    return nn.Sequential(*layers)

class VAE_Discriminator(nn.Module):
    def __init__(self, zDim=zDim):
        super(VAE_Discriminator, self).__init__()
        discriminationOutputFeatures = 128
        
        self.residualBlock1 = residualBlockDownFullDisc(3, block_outdim[0], blocks=blocks[0], downsample=2) #256x32x32
        self.residualBlock2 = residualBlockDownFullDisc(block_outdim[0], block_outdim[1], blocks=blocks[1], downsample=2) #512x16x16
        self.residualBlock3 = residualBlockDownFullDisc(block_outdim[1], block_outdim[2], blocks=blocks[2], downsample=2) #1024x8x8
        self.residualBlock4 = nn.Sequential(residualBlockDownFullDisc(block_outdim[2], block_outdim[3], blocks=blocks[3], downsample=2), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        
        self.blockSequence = torch.nn.Sequential(self.residualBlock1, self.residualBlock2, self.residualBlock3, self.residualBlock4)
        self.encFC = nn.Linear(block_outdim[-1], 1)

    def forward(self, x, feature_matching=False):
        x = self.blockSequence(x)
        if feature_matching:
            return x
        x = self.encFC(x)
        return x

class Encoder(nn.Module):
    def __init__(self, imgChannels=3, zDim=zDim):
        super(Encoder, self).__init__()
        #self.residualBlock1 = nn.Sequential(nn.Conv2d(3, block_outdim[0], kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2)) #64x64x64
        self.residualBlock1 = residualBlockDownFullDisc(3, block_outdim[0], blocks=blocks[0], downsample=2) #256x32x32
        self.residualBlock2 = residualBlockDownFullDisc(block_outdim[0], block_outdim[1], blocks=blocks[1], downsample=2) #512x16x16
        self.residualBlock3 = residualBlockDownFullDisc(block_outdim[1], block_outdim[2], blocks=blocks[2], downsample=2) #1024x8x8
        self.residualBlock4 = nn.Sequential(residualBlockDownFullDisc(block_outdim[2], block_outdim[3], blocks=blocks[3], downsample=2))
        
        self.blockSequence = torch.nn.Sequential(self.residualBlock1, self.residualBlock2, self.residualBlock3, self.residualBlock4)
        self.momentCreation = torch.nn.Conv2d(block_outdim[-1], 2*block_outdim[-1], 1)

    def encoder(self, x):
        x = self.blockSequence(x)
        moments = self.momentCreation(x)
        return DiagonalGaussianDistribution(moments)

    def forward(self, x, sampling=True):
        dist = self.encoder(x)
        return dist

def residualBlockUp(channelIn, channelOut, blocks=3, upsample=2):
    layers = []
    for i in range(blocks):
        filters = [channelIn, channelIn, channelIn, channelOut] if (i == 0) else [channelOut, channelIn, channelIn, channelOut]
        kernels = [1, 4, 1] if (i == 0) else [1,3,1]
        strides = [1, upsample, 1] if (i == 0) else [1, 1, 1]
        paddings = [0, 1, 0]
        shortcut = nn.ConvTranspose2d(channelIn, channelOut, kernel_size=1, stride=upsample, bias=False, padding=0, output_padding=1) if (i == 0) else nn.Identity()
        layer = torchgan.layers.ResidualBlockTranspose2d(filters=filters, kernels=kernels, strides=strides, paddings=paddings, 
                                           nonlinearity=nn.LeakyReLU(0.2), batchnorm=False, 
                                           shortcut=shortcut, 
                                           last_nonlinearity=nn.LeakyReLU(0.2))
        layers = layers + [layer]
    return nn.Sequential(*layers)
    
class Decoder(nn.Module):
    def __init__(self, imgChannels=3, zDim=zDim):
        super(Decoder, self).__init__()
        
        self.residualBlock1 = residualBlockUp(block_outdim[3], block_outdim[2], blocks=blocks[3], upsample=2)
        self.residualBlock2 = residualBlockUp(block_outdim[2], block_outdim[1], blocks=blocks[2], upsample=2) #256x32x32
        self.residualBlock3 = residualBlockUp(block_outdim[1], block_outdim[0], blocks=blocks[1], upsample=2) #512x16x16
        self.residualBlock4 = residualBlockUp(block_outdim[0], 3, blocks=blocks[0], upsample=2) #1024x8x8
        
        self.blockSequence = torch.nn.Sequential(self.residualBlock1, self.residualBlock2, self.residualBlock3, self.residualBlock4)

    def forward(self, x):
        x = self.blockSequence(x)
        return x

class AutoencoderModel(pl.LightningModule):
    def __init__(self, in_size, in_size_sqrt, img_depth):
        super().__init__()
        self.in_size = in_size
        self.in_size_sqrt = in_size_sqrt
        self.img_depth = img_depth
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = VAE_Discriminator()

    def forward(self, x, sampling=True):
        encoding = self.encoder(x)
        if sampling:
            return self.decoder(encoding.sample())
        else:
            return self.decoder(encoding.mode())
            
    def get_encoder_loss(self, imgs, batch_size, prefix):
        encoding = self.encoder(imgs)
        G_imgs = self.decoder(encoding.sample())
        D_imgs_activation = self.discriminator(imgs, feature_matching=True)
        D_decoder_activation = self.discriminator(G_imgs, feature_matching=True)
    
        kl_divergence = encoding.kl().mean()
        like_loss = torch.nn.functional.mse_loss(D_imgs_activation, D_decoder_activation)
        encoder_loss = kl_divergence + like_loss
        self.log(prefix+"encoder_kl_loss", kl_divergence)
        self.log(prefix+"encoder_like_loss", like_loss)
        return encoder_loss
        
    def get_decoder_loss(self, imgs, batch_size, prefix):
        encoding = self.encoder(imgs)
        sample = encoding.sample()
        fake_mean = torch.randn_like(sample)
        G_imgs = self.decoder(sample)
        G_fake = self.decoder(fake_mean)
        
        D_imgs_activation = self.discriminator(imgs, feature_matching=True)
        D_decoder_activation = self.discriminator(G_imgs, feature_matching=True)
        D_decoder_logit = self.discriminator(G_imgs)
        D_fake_logit = self.discriminator(G_fake)
        
        decoder_loss_encoded = torch.mean(D_decoder_logit)
        decoder_loss_fake = torch.mean(D_fake_logit)
        like_loss = torch.nn.functional.mse_loss(D_imgs_activation, D_decoder_activation)
        decoder_loss = decoder_loss_encoded + decoder_loss_fake + like_loss
        return decoder_loss
        
    def get_discriminator_loss(self, imgs, batch_size, prefix):
        encoding = self.encoder(imgs)
        sample = encoding.sample()
        fake_mean = torch.randn_like(sample)
        G_imgs = self.decoder(sample)
        G_fake = self.decoder(fake_mean)
        
        D_imgs_logit = self.discriminator(imgs)
        D_decoder_logit = self.discriminator(G_imgs)
        D_fake_logit = self.discriminator(G_fake)
        
        discriminator_loss_imgs = torch.mean(D_imgs_logit)
        discriminator_loss_decoder = -0.5 * torch.mean(D_decoder_logit)
        discriminator_loss_fake = -0.5 * torch.mean(D_fake_logit)
        #GP loss
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(imgs)
        alpha = alpha.to(device=self.device)
        interpolated = alpha * imgs.data + (1 - alpha) * G_imgs.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()
        # Calculate rating of interpolated samples
        with torch.enable_grad():
            D_interp_logit = self.discriminator.forward(interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=D_interp_logit, inputs=interpolated,
                                   grad_outputs=torch.ones_like(D_interp_logit),
                                   create_graph=True, retain_graph=True)[0]
        
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        # Return gradient penalty
        discriminator_loss_gp = ((gradients_norm - 1) ** 2).mean()    
            
        discriminator_loss = discriminator_loss_imgs + discriminator_loss_decoder + discriminator_loss_fake + discriminator_loss_gp
        return discriminator_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_size = batch.shape[0]
        imgs = batch
        if (optimizer_idx == 0):
            encoder_loss = self.get_encoder_loss(imgs, batch_size, "train/")
            self.log("train/encoder_loss", encoder_loss)
            return encoder_loss
        elif (optimizer_idx == 1):
            decoder_loss = self.get_decoder_loss(imgs, batch_size, "train/")
            self.log("train/decoder_loss", decoder_loss)
            return decoder_loss
        elif (optimizer_idx == 2):
            discriminator_loss = self.get_discriminator_loss(imgs, batch_size, "train/")
            self.log("train/discriminator_loss", discriminator_loss)
            return discriminator_loss
        return

    def validation_step(self, batch, batch_idx):
        batch_size = batch.shape[0]
        encoder_loss = self.get_encoder_loss(batch, batch_size, "val/")
        decoder_loss = self.get_decoder_loss(batch, batch_size, "val/")
        discriminator_loss = self.get_discriminator_loss(batch, batch_size, "val/")
        self.log("val/encoder_loss", encoder_loss)
        self.log("val/decoder_loss", decoder_loss)
        self.log("val/discriminator_loss", discriminator_loss)
        if (batch_idx == 0):
            self.val_batch = batch
        return
    
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
        encoding = self.encoder(self.val_batch)
        decoding = self.decoder(encoding.mode())
        x = torch.cat([self.val_batch, decoding], dim=0)
        x = (x.clamp(-1, 1) + 1) / 2.0

        tb_logger.add_images(f"val/output_images", x, self.current_epoch)
        print("Images added!")

    def configure_optimizers(self):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=2e-6)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=2e-6)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-6)
        return [encoder_optimizer, decoder_optimizer, discriminator_optimizer], []
 