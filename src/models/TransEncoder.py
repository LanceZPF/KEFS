##################################################################
# TransEncoder
# TransDecoder: 
### First try use a trans block to make the final output and then a 3*3 conv
### Following ViT GAN, modified v3 to mlp and then output the output without 3*3 conv
##################################################################
import torch.nn as nn
import torch
from functools import partial
from src.weight_init import trunc_normal_, lecun_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels= [24, 48, 96, 192], dropout = 0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels[0],
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        
        self.down1 = Downsample(out_channels[0], out_channels[1])
        
        self.conv2 = torch.nn.Conv2d(out_channels[1],
                                     out_channels[2],
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[2])
        
        self.down2 = Downsample(out_channels[2], out_channels[3])
        
        self.conv3 = torch.nn.Conv2d(out_channels[3],
                                     out_channels[3],
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)        
        
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        h = x
        
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.down1(h)

#         h = self.dropout(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.down2(h)

        h = self.conv3(h)

        return h
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class OutBlock(nn.Module):

    def __init__(self, dim, num_heads, out_dim, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features = out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x
    
def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
        
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransEncoder(nn.Module):
    "A transformer encoder"
    def __init__(self, embed_dim=300, out_chans=64, depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()

        # self.down = DownBlock(in_channels = in_chans)
        self.out_conv = torch.nn.Linear(embed_dim, out_chans)

        # then for transformers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        # num_patches = int(img_size / 4) ** 2
        #
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.init_weights(weight_init)
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def forward(self, x):
        # input x: B, C, H, W
        # x = self.down(x) # out: B, C, _H, _W
        # B, C, _ = x.shape
        # x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1).contiguous() # reshape to: B, 1, C
        x = self.blocks(x)
        x = self.norm(x)
        x = self.out_conv(x)
        x = x.permute(0, 2, 1).contiguous()  # reshape to: B, C, 1
        return x

    
class TransEncoder_rectangle(nn.Module):
    "A transformer encoder"
    def __init__(self, img_size=[192, 96], in_chans=3, out_chans = 64, embed_dim=192, depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        
        # first init a downsample block
        self.down = DownBlock(in_channels = in_chans)
        self.out_conv = torch.nn.Conv2d(embed_dim,
                                        out_chans,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)  
        
        # then for transformers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        num_patches = int(img_size[0] / 4) * int(img_size[1] / 4)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.init_weights(weight_init)
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def forward(self, x):
        # input x: B, C, H, W
        x = self.down(x) # out: B, C, _H, _W
        B, C, _H, _W = x.shape
        x = x.reshape(B, C, _H*_W).permute(0, 2, 1).contiguous() # reshape to: B, _H * _W, C
#             x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(B, C, _H, _W).contiguous()
        x = self.out_conv(x)
        return x
    
    
# class TransDecoder(nn.Module):
#     "A trivial transformer decoder with 3 * 3 conv at last"
#     def __init__(self, img_size=128, in_chans=64, out_chans = 3, embed_dim=192, depth=6,
#                  num_heads=4, mlp_ratio=4., qkv_bias=True,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
#                  act_layer=None, weight_init=''):
#         super().__init__()
        
#         self.in_conv = torch.nn.Conv2d( in_chans,
#                                         embed_dim,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)
        
#         # transformers for decoding
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
        
#         self.f = 4
#         num_patches = int(img_size / self.f) ** 2
        
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth+1)]  # stochastic depth decay rule
        
#         self.blocks = nn.Sequential(*[
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)])
        
        
#         self.out_chans = out_chans
#         self.out = OutBlock(
#                 dim=embed_dim, num_heads=num_heads, out_dim = self.f ** 2 * out_chans, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[depth], norm_layer=norm_layer, act_layer=act_layer)
        
#         self.norm = norm_layer(self.f ** 2 * out_chans)
#         self.out_conv = torch.nn.Conv2d(out_chans,
#                                         out_chans,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
        
#         self.init_weights(weight_init)
        
#     def init_weights(self, mode=''):
#         assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
#         head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
#         trunc_normal_(self.pos_embed, std=.02)
#         if mode.startswith('jax'):
#             # leave cls token as zeros to match jax impl
#             named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
#         else:
#             self.apply(_init_vit_weights)

#     def forward(self, x):
#         # input x: B, C, _H, _W
#         x = self.in_conv(x)
#         B, C, _H, _W = x.shape
#         x = x.reshape(B, C, _H*_W).permute(0, 2, 1) # reshape to: B, _H * _W, C
# #             x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.out(x)
#         x = self.norm(x) # B, _H * _W, 3 * 16
#         x = x.reshape(B, _H, _W, self.f, self.f, self.out_chans)
#         x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, _H * self.f, _W * self.f)
#         x = self.out_conv(x)
#         return x
    
class TransDecoder(nn.Module):
    "A pure ViT decoder"
    def __init__(self, img_size=128, in_chans=64, out_chans = 3, embed_dim=192, depth=6,
                 num_heads=4, mlp_ratio=4., use_pos_embed = False, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        
        self.in_conv = torch.nn.Conv2d( in_chans,
                                        embed_dim,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
        # transformers for decoding
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.f = 4
        num_patches = int(img_size / self.f) ** 2
        
        self.use_pos_embed = use_pos_embed
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth+1)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.out_chans = out_chans
        self.out = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features = self.f ** 2 * out_chans, act_layer=act_layer)
        
        self.init_weights(weight_init)
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def forward(self, x):
        # input x: B, C, _H, _W
        x = self.in_conv(x)
        B, C, _H, _W = x.shape
        x = x.reshape(B, C, _H*_W).permute(0, 2, 1) # reshape to: B, _H * _W, C
        
        if self.use_pos_embed:
            x = self.pos_drop(x + self.pos_embed)
            
        x = self.blocks(x)
        x = self.norm(x) # B, _H * _W, 3 * 16
        x = self.out(x)
        x = x.reshape(B, _H, _W, self.f, self.f, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, _H * self.f, _W * self.f)
        return x
