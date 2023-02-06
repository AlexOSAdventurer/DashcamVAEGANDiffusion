import os
import sys
import yaml

sys.path.append(__path__[0])

from . import distribution
from . import model

autoencoder_checkpoint_path = os.path.join(__path__[0], "autoencoderkl.ckpt")
autoencoder_config_path = os.path.join(__path__[0], "autoencoder_kl_256x256x3.yaml")

config_data = yaml.safe_load(open(autoencoder_config_path))

def generate_pretrained_model():
    return model.AutoencoderKL.load_from_checkpoint(autoencoder_checkpoint_path, ddconfig=config_data['model']['params']['ddconfig'],
                 lossconfig=config_data['model']['params']['lossconfig'],
                 embed_dim=config_data['model']['params']['embed_dim'],
                 base_learning_rate=config_data['model']['base_learning_rate'])

__all__ = ["distribution", "model", "generate_pretrained_model", "config_data"]