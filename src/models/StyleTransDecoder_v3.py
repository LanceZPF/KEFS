"""
A localViT decoder with prototypical Style Injection
Written by Xuanchi REN, 7.18
v3 feature: 6 * (style mixer + 1 * Trans Block)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from src.modules.ViT.ViT_helper import DropPath, Mlp, trunc_normal_, _init_vit_weights
from src.modules.ViT.localvit import Block

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerDecoderLayer_QKV(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", mode= None):
        super(TransformerDecoderLayer_QKV, self).__init__()
        self.mode = mode
        if mode == 'self_attn':
            self.self_norm = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer_QKV, self).__setstate__(state)

    def forward(self, tgt, prototype, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attn=False):
        """
        use self-attention as very beginning (after vq)
        cross attention can use additional prototype as key.
        assert input: tgt: T, B, C; prototype: T, B, C; 
        """
        attn_maps = {}

        if self.mode == 'self_attn':
            tgt = self.self_norm(tgt)
            tgt2, attn_map = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
            if return_attn:
                attn_maps['self_attn'] = attn_map.detach().cpu().numpy()
            
            tgt = tgt + self.dropout1(tgt2)
            
        tgt = self.norm1(tgt)
        tgt2, attn_map = self.multihead_attn(tgt, prototype, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        if return_attn:
            attn_maps['cross_attn'] = attn_map.detach().cpu().numpy()
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt) no need to norm here
    
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
        return tgt, attn_maps

    
class PrototypicalStyleDecoder(nn.Module):
    """
    embed_dim: channel of spatial
    embed_dim_style: channel of input style bank
    """
    def __init__(self, n_layers, embed_dim_style, embed_dim, nhead, dim_feedforward, dropout=0.1, activation="relu", mode = None):
        super(PrototypicalStyleDecoder, self).__init__()
        
        self.style_project = nn.Linear(embed_dim_style, embed_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerDecoderLayer_QKV(embed_dim, nhead, dim_feedforward, dropout, activation, mode))

    def forward(self, x, prototype, spk_bank, return_attn=False, padding_mask=None):
        """
        :param x: need T x B x C while input B, T, C
        :param spk_bank: T x B x C
        :return: T x B x C
        """
        # first inject project style
        x = x.permute(1,0,2) # to not batch first
        spk_bank = self.style_project(spk_bank)
        _, B, _ = x.shape
        output = x
        attn_maps = {}
        for idx, m in enumerate(self.layers):
            output, attn_map = m(output, prototype, spk_bank, return_attn=return_attn, tgt_key_padding_mask=padding_mask)
            attn_maps[idx] = attn_map
            
        output = output.permute(1,0,2) # to batch first
        return output, attn_maps
    
class StyleTransDecoder(nn.Module):
    "A localViT decoder with prototypical Style Injection"
    def __init__(self, in_chans=64, in_chans_style=64, out_chans=1, embed_dim=192,
                 n_emb_style=512, depth=6, num_heads=4, mlp_ratio=4., use_pos_embed = False, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None, 
                 weight_init='', act=2, reduction=4, wo_dp_conv=False, dp_first=False):
        super().__init__()
        
        # parse act
        if act == 1:
            act = 'relu6'
        elif act == 2:
            act = 'hs'
        elif act == 3:
            act = 'hs+se'
        elif act == 4:
            act = 'hs+eca'
        else:
            act = 'hs+ecah'
        # parse norm and info
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU # for out MLP
        
        self.f = 4
        # num_patches = int(img_size / self.f) ** 2
        
        self.use_pos_embed = use_pos_embed # False
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # in conv
        self.in_conv = torch.nn.Linear(in_chans,
                                        embed_dim)
        
        # first style inject with self attention
        # first style block
        self.prototype = nn.Parameter(torch.randn(n_emb_style, embed_dim), requires_grad=True)
        
        self.style_mixer1 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio), mode = 'self_attn')
        self.block1 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )
        
        # second style block
        self.style_mixer2 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio))
        self.block2 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )
        # third style block
        self.style_mixer3 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio))
        self.block3 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )
        # fourth style block
        self.style_mixer4 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio))
        self.block4 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )
        # fifth style block
        self.style_mixer5 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio))
        self.block5 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[4], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )
        # sixth style block
        self.style_mixer6 = PrototypicalStyleDecoder(n_layers=1, embed_dim_style=in_chans_style, embed_dim=embed_dim,
                                                     nhead=num_heads, dim_feedforward = int(embed_dim * mlp_ratio))
        self.block6 = Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[5], norm_layer=norm_layer, act=act, 
                        reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
                           )

        # prepare for output RGB
        self.norm = norm_layer(embed_dim)
        
        self.out_chans = out_chans
        self.out = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features = out_chans, act_layer=act_layer)
        
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

    def forward(self, x, spk_bank):
        # input x: 1, B, C
        # input spk_bank: 1, B, C

        # x = x.squeeze(2)
        # x = self.in_conv(x)
        # x = x.unsqueeze(-1)

        _, B, C = x.shape
        # x = x.permute(1, 0, 2) # reshape to: B, 1, newC

        # if self.use_pos_embed:
        #     x = self.pos_drop(x + self.pos_embed)

        spk_bank = self.in_conv(spk_bank)
        
        prototype = self.prototype.unsqueeze(1).repeat((1, B, 1)) # 1, B, newC
        prototype = prototype.permute(1, 0, 2) # 1, B, newC -> B, 1, newC

        # print(prototype.shape)
        # print(spk_bank.shape)
        # print(x.shape)

        # first block
        prototype, _ = self.style_mixer1(prototype, spk_bank, x)
        prototype = self.block1(prototype)
        # second block
        prototype, _ = self.style_mixer2(prototype, spk_bank, x)
        prototype = self.block2(prototype)
        # thrid block
        prototype, _ = self.style_mixer3(prototype, spk_bank, x)
        prototype = self.block3(prototype)
        # fourth style block
        prototype, _ = self.style_mixer4(prototype, spk_bank, x)
        prototype = self.block4(prototype)
        # fifth style block
        prototype, _ = self.style_mixer5(prototype, spk_bank, x)
        prototype = self.block5(prototype)
        # sixth style block
        prototype, _ = self.style_mixer6(prototype, spk_bank, x)
        prototype = self.block6(prototype)

        x = prototype

        # # first block
        # x, _ = self.style_mixer1(x, prototype, spk_bank)
        # x = self.block1(x)
        # # second block
        # x, _ = self.style_mixer2(x, prototype, spk_bank)
        # x = self.block2(x)
        # # thrid block
        # x, _ = self.style_mixer3(x, prototype, spk_bank)
        # x = self.block3(x)
        # # fourth style block
        # x, _ = self.style_mixer4(x, prototype, spk_bank)
        # x = self.block4(x)
        # # fifth style block
        # x, _ = self.style_mixer5(x, prototype, spk_bank)
        # x = self.block5(x)
        # # sixth style block
        # x, _ = self.style_mixer6(x, prototype, spk_bank)
        # x = self.block6(x)

        # prepare for RGB
        x = self.norm(x) # B, _H * _W, 3 * 16
        x = self.out(x)

        x = x.squeeze(1)

        # x = x.reshape(B, _H, _W, self.f, self.f, self.out_chans)
        # x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, _H * self.f, _W * self.f)
        return x