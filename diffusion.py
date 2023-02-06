import torch
import numpy as np
import math

def _beta(current_t, max_t, beta_small, beta_large):
    return beta_small + (current_t / max_t) * (beta_large - beta_small)

def _alpha(current_t, max_t, beta_small, beta_large):
    return 1.0 - _beta(current_t, max_t, beta_small, beta_large)

def _alpha_bar(current_t, max_t, beta_small, beta_large):
    return math.prod([_alpha(current_t, max_t, beta_small, beta_large) for j in range(current_t)])

def create_random_time_steps(number_of_items, max_t, device):
    return torch.from_numpy(np.random.choice(max_t, size=(number_of_items,), p=np.ones([max_t])/max_t)).to(device)
    
def diffuse_images(images, time_steps, max_t, beta_small, beta_large):
    noised_images = torch.empty_like(images)
    source_noise = torch.randn_like(images)
    for i in range(images.shape[0]):
        current_alpha_bar = _alpha_bar(float(time_steps[i]), max_t, beta_small, beta_large)
        noised_images[i] = (math.sqrt(current_alpha_bar) * images[i]) + (math.sqrt(1.0 - current_alpha_bar) * source_noise[i])
    
    return noised_images, source_noise

def estimate_noise(ddim_model, noised_images, time_steps, z_sem):
    return ddim_model.forward(noised_images, time_steps, z_sem)
    
def encode_semantic(semantic_encoder, images):
    return semantic_encoder.forward(images)
