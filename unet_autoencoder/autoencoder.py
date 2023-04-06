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
    return torch.nn.Dropout(p=0.1)
    
def skip_connection(input_channels, output_channels):
    return torch.nn.Identity() if (input_channels == output_channels) else conv_fc(input_channels, output_channels)
    
def conv_nd(input_channels, output_channels):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    
def conv_fc(input_channels, output_channels):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
    
def activation():
    return torch.nn.SiLU()
    
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
    
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class UNetBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_list = torch.nn.Sequential(
            UNetLayerNormalization(input_channels),
            activation(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels),
            activation(),
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
            activation(),
            torch.nn.Linear(self.semantic_latent_channels, output_channels)
        )
        self.timestep_mlp = torch.nn.Sequential(
            activation(),
            torch.nn.Linear(self.timestep_dims, 2 * output_channels)
        )
        self.block_list_1 = torch.nn.Sequential(
            UNetLayerNormalization(input_channels),
            activation(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels))
        self.block_list_2 = torch.nn.Sequential(
            activation(),
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

class AttentionBlock(torch.nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = UNetLayerNormalization(channels)
        self.qkv = conv_fc(channels, channels * 3)
        self.attention = QKVAttentionDiffAE(self.num_heads)

        self.proj_out = zero_module(conv_fc(channels, channels))

    def forward(self, x):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, c * 3, -1)
        h = self.attention(qkv).reshape(b, c, *spatial)
        h = self.proj_out(h)
        return x + h
        
class QKVAttentionDiffAE(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        batch_size, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(batch_size * self.num_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch)) # Scale is ch^0.25, not ch^0.5. Not sure why, probably has to do with some literature I am not familiar with
        # My understanding is that this stuff is verbatim from Attention is all you need
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(batch_size, -1, length)

class UNetBlockGroup(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_res_blocks, upsample=False, upsample_target_channels = None, downsample=False, conditional=False, num_heads=None):
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
        for i in range(self.num_res_blocks):
            self.block_list.append(block_type(output_channels, output_channels))
            if (num_heads is not None):
                self.block_list.append(AttentionBlock(output_channels, num_heads))
        if upsample:
            self.upsample_conv = conv_nd(self.output_channels, self.upsample_target_channels)
        
    def forward(self, x, t = None, z_sem = None, return_unscaled_output=False):
        if self.conditional:
            for module in self.block_list:
                if isinstance(module, AttentionBlock):
                    x = module(x)
                else:
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
        self.attention_heads = [16]
        self.attention_resolutions = [16]
        
        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )
        
        current_resolution = self.input_size
        previous_channels = self.block_list_base_channels
        for entry in self.block_list_channels_mult:
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            self.block_list.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=False, downsample=True, conditional=False, num_heads=current_heads))
            previous_channels = current_channels
            current_resolution = current_resolution // 2
        
        self.final_output_module = torch.nn.Sequential(
            UNetLayerNormalization(previous_channels),
            activation(),
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
        self.attention_heads = [16]
        self.attention_resolutions = [16]
        
        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )
        self.lastSide = torch.nn.Sequential(
            conv_nd(self.block_list_base_channels, self.input_channels)
        )
        self.downSide = torch.nn.ModuleList([])
        self.upSide = torch.nn.ModuleList([])
        
        current_resolution = self.input_size
        previous_channels = self.block_list_base_channels
        for entry in self.block_list_channels_mult:
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            self.downSide.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=False, downsample=True, conditional=True, num_heads=current_heads))
            previous_channels = current_channels
            current_resolution = current_resolution // 2
        current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
        self.middleModule = UNetBlockGroup(previous_channels, previous_channels * 2, self.num_res_blocks, upsample=True, upsample_target_channels = previous_channels, downsample=False, conditional=True, num_heads=current_heads)
        current_resolution = current_resolution * 2
        previous_channels = previous_channels * 2
        block_list_channels_mult_reversed = self.block_list_channels_mult[::-1]
        for i in range(len(block_list_channels_mult_reversed) - 1):
            entry = block_list_channels_mult_reversed[i]
            next_entry = block_list_channels_mult_reversed[i + 1]
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            next_channels = self.block_list_base_channels * next_entry
            self.upSide.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=True, upsample_target_channels = next_channels, downsample=False, conditional=True, num_heads=current_heads))
            current_resolution = current_resolution * 2
            previous_channels = next_channels * 2
        current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
        self.upSide.append(UNetBlockGroup(previous_channels, self.block_list_base_channels, self.num_res_blocks, upsample=False, downsample=False, conditional=True, num_heads=current_heads))
    
    def forward(self, x, t=None, cond=None):
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
        
class DDIM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_skip_net = MLPSkipNet()
        
    def forward(self, x, t):
        return self.mlp_skip_net(x, t)

class MLPSkipNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 512
        self.num_hidden_channels = 2048
        self.num_time_layers = 2
        self.num_time_emb_channels = 64
        self.num_condition_bias = 1
        self.num_regular_layers = 10
        
        layers = []
        in_channels = self.num_time_emb_channels
        out_channels = self.num_channels
        for i in range(self.num_time_layers):
            layers.append(nn.Linear(in_channels, out_channels))
            if (i != (self.num_time_layers - 1)):
                layers.append(activation())
            in_channels = out_channels
        self.time_embed = torch.nn.Sequential(*layers)
        
        self.regular_layers = torch.nn.ModuleList([])
        in_channels = self.num_channels
        out_channels = self.num_hidden_channels
        for i in range(self.num_layers):
            if (i == (self.num_layers - 1)):
                self.layers.append(MLPBlock(in_channels, self.num_channels, norm=False, cond=False, act=False, cond_channels=self.num_channels, cond_bias=self.num_condition_bias))
            else:
                self.layers.append(MLPBlock(in_channels, out_channels, norm=True, cond=True, act=True))
            in_channels = out_channels + self.num_channels
    
    def forward(self, x, t):
        t = timestep_embedding(t, self.num_time_emb_channels)
        t_cond = self.time_embed(t)
        h = x
        for i in range(self.num_regular_layers):
            if (i != 0):
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(h, cond=t_cond)
        return h
        
class MLPBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm, cond, act, cond_channels=None, cond_bias=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = torch.nn.LayerNorm(out_channels) if norm else torch.nn.Identity()
        self.act = activation() if act else torch.nn.Identity()
        self.use_cond = cond
        self.cond_channels = cond_channels
        self.cond_bias = cond_bias
        
        self.linear = torch.nn.Linear(self.in_channels, self.out_channels)
        if self.use_cond:
            self.linear_emb = torch.nn.Linear(self.cond_channels, self,out_channels)
            self.cond_layers = torch.nn.Sequential(self.act, self.linear_emb)
            
    def forward(self, x, cond=None):
        x = self.linear(x)
        if (self.use_cond):
            cond = self.cond_layers(cond)
            x = x * (self.cond_bias + cond)
        x = self.norm(x)
        x = self.act(x)
        return x