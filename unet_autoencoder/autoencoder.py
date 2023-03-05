import torch
import torch.nn.functional as F
import math

# We assume channels can be cleanly divided by 32
def UNetLayerNormalization(channels):
    return torch.nn.GroupNorm(32, channels)
    
def Upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

def Downsample(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)
    
def dropout():
    return torch.nn.Identity()
    
def skip_connection(input_channels, output_channels):
    return torch.nn.Identity() if (input_channels == output_channels) else conv_fc(input_channels, output_channels)
    
def conv_nd(input_channels, output_channels):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    
def conv_fc(input_channels, output_channels):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
    
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class UNetBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_list = torch.nn.Sequential(
            UNetLayerNormalization(input_channels),
            torch.nn.SiLU(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels),
            torch.nn.SiLU(),
            dropout(),
            conv_nd(output_channels, output_channels)
        )
        self.skip_connection = skip_connection(input_channels, output_channels)
        
    def forward(self, x):
        return self.block_list(x) + self.skip_connection(x)
        
class UNetBlockConditional(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.semantic_latent_channels = 512
        self.timestep_dims = 512
        self.semantic_affine = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.semantic_latent_channels, output_channels)
        )
        self.timestep_mlp = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.timestep_dims, 2 * output_channels)
        )
        self.block_list_1 = torch.nn.Sequential(
            UNetLayerNormalization(input_channels),
            torch.nn.SiLU(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels))
        self.block_list_2 = torch.nn.Sequential(
            torch.nn.SiLU(),
            dropout(),
            conv_nd(output_channels, output_channels)
        )
        self.skip_connection = skip_connection(input_channels, output_channels)
        
    def forward(self, x, t, z_sem):
        mid_point = self.block_list_1(x)
        t_emb = self.timestep_mlp(timestep_embedding(t, self.timestep_dims))
        t_s, t_b = torch.chunk(torch.unsqueeze(torch.unsqueeze(t_emb, dim=-1), dim=-1), 2, dim=1)
        z_sem_scaling = torch.unsqueeze(torch.unsqueeze(self.semantic_affine(z_sem), dim=-1), dim=-1)
        conditioned = t_s * mid_point
        conditioned = conditioned + t_b
        conditioned = z_sem_scaling * conditioned
        final_point = self.block_list_2(conditioned)
        skipped = self.skip_connection(x)
        return final_point + skipped

class UNetBlockGroup(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_res_blocks, upsample=False, upsample_target_channels = None, downsample=False, conditional=False):
        super().__init__()
        block_type = UNetBlockConditional if conditional else UNetBlock
        self.block_list = torch.nn.ModuleList([])
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_res_blocks = num_res_blocks
        self.upsample = upsample
        self.upsample_target_channels = upsample_target_channels
        self.downsample = downsample
        self.conditional = conditional
        self.block_list.append(block_type(input_channels, output_channels))
        for i in range(self.num_res_blocks - 1):
            self.block_list.append(block_type(output_channels, output_channels))
        if upsample:
            self.upsample_conv = conv_nd(self.output_channels, self.upsample_target_channels)
        
    def forward(self, x, t = None, z_sem = None, return_unscaled_output=False):
        if self.conditional:
            for module in self.block_list:
                x = module(x, t, z_sem)
        else:
            for module in self.block_list:
                x = module(x)

        assert (not (self.upsample and self.downsample)), "You can't be both upsampling and downsampling!"
        if self.upsample:
            res = self.upsample_conv(Upsample(x))
            if return_unscaled_output:
                x = (res, x)
            else:
                x = res
        if self.downsample:
            res = Downsample(x)
            if return_unscaled_output:
                x = (res, x)
            else:
                x = res
        return x

class UNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_list = torch.nn.ModuleList([])
        self.input_size = 64
        self.input_channels = 3
        self.block_list_base_channels = 64
        self.block_list_channels_mult = [1, 2, 4, 8, 8]
        self.latent_space = 512
        self.num_res_blocks = 2
        
        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )
        
        previous_channels = self.block_list_base_channels
        for entry in self.block_list_channels_mult:
            current_channels = self.block_list_base_channels * entry
            self.block_list.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=False, downsample=True, conditional=False))
            previous_channels = current_channels
        
        self.final_output_module = torch.nn.Sequential(
            UNetLayerNormalization(previous_channels),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            conv_fc(previous_channels, self.latent_space),
            torch.nn.Flatten()
        )
        
    def forward(self, x):
        x = self.firstSide(x)
        for module in self.block_list:
            x = module(x)
        x = self.final_output_module(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 64
        self.input_channels = 3
        self.block_list_base_channels = 64
        self.block_list_channels_mult = [1, 2, 4, 8]
        self.num_res_blocks = 2
        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )
        self.lastSide = torch.nn.Sequential(
            conv_nd(self.block_list_base_channels, self.input_channels)
        )
        self.downSide = torch.nn.ModuleList([])
        self.upSide = torch.nn.ModuleList([])
        
        previous_channels = self.block_list_base_channels
        for entry in self.block_list_channels_mult:
            current_channels = self.block_list_base_channels * entry
            self.downSide.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=False, downsample=True, conditional=True))
            previous_channels = current_channels
        self.middleModule = UNetBlockGroup(previous_channels, previous_channels * 2, self.num_res_blocks, upsample=True, upsample_target_channels = previous_channels, downsample=False, conditional=True)
        previous_channels = previous_channels * 2
        block_list_channels_mult_reversed = self.block_list_channels_mult[::-1]
        for i in range(len(block_list_channels_mult_reversed) - 1):
            entry = block_list_channels_mult_reversed[i]
            next_entry = block_list_channels_mult_reversed[i + 1]
            current_channels = self.block_list_base_channels * entry
            next_channels = self.block_list_base_channels * next_entry
            self.upSide.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=True, upsample_target_channels = next_channels, downsample=False, conditional=True))
            previous_channels = current_channels
        self.upSide.append(UNetBlockGroup(previous_channels, self.block_list_base_channels, self.num_res_blocks, upsample=False, downsample=False, conditional=True))
    
    def forward(self, x, t=None, cond=None):
        if t is None:
            t = torch.randn((x.shape[0])).to(x.device)
        if cond is None:
            cond = torch.randn((x.shape[0], 512)).to(x.device)
        x = self.firstSide(x)
        intermediate_outputs = []
        for module in self.downSide:
            x = module(x, t, cond, return_unscaled_output=True)
            intermediate_outputs.append(x[1])
            x = x[0]
        x = self.middleModule(x, t, cond)
        for module in self.upSide:
            x = torch.cat([x, intermediate_outputs.pop()], dim=1)
            x = module(x, t, cond)
        x = self.lastSide(x)
        return x

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.encoder = UNetEncoder()
    
    def forward(self, x, t, cond):
        return self.unet(x, t, cond)