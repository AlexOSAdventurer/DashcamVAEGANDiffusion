import os
import sys
from . import autoencoder

    
def generate_model():
    model = autoencoder.Autoencoder()
    return model

__all__ = ['generate_model']