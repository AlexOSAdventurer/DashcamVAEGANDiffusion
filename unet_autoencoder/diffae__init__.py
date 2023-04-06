import os
import sys

def add_module_to_path(path_to_add=__path__[0]):
    sys.path.append(path_to_add)

def remove_module_from_path():
    sys.path.pop()

add_module_to_path(os.path.join(__path__[0], 'diffae/'))

import templates

def dashcam64_autoenc():
    conf = templates.autoenc_base()
    conf.data_name = 'berkeleydeepdrive'
    conf.warmup = 0
    conf.batch_size = 16
    conf.total_samples = 70000
    conf.net_ch = 64
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_num_res_blocks = 2
    conf.eval_every_samples = 70000
    conf.eval_ema_every_samples = 70000
    conf.lr = 1e-4
    conf.scale_up_gpus(2, num_nodes=4)
    conf.make_model_conf()
    return conf
    
def generate_model():
    add_module_to_path(os.path.join(__path__[0], 'diffae/'))
    conf = dashcam64_autoenc()
    model = conf.make_model_conf().make_model()
    remove_module_from_path()
    
    return model

remove_module_from_path()

__all__ = ['generate_model']
