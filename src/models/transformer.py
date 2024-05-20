import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.modules.disentangle.ortho_utils import torch_expm

class StaticSelfAttn(nn.Module):
    def __init__(self, n_pos, d_model, nhead):
        super(StaticSelfAttn, self).__init__()
        # Kaiming Initialize
        self.weight = nn.Parameter(torch.randn(1, 1, nhead, n_pos, n_pos) * np.sqrt(2 / n_pos), requires_grad=True)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_model = d_model
        assert self.d_model % self.nhead == 0

    def forward(self, x):
        """
        :param x: T x B x C
        :return: T x B x C
        """
        T, B, C = x.shape

        x = self.value_proj(x).reshape(T, B, self.nhead, C//self.nhead)
        x = x.permute((3, 1, 2, 0)).unsqueeze(-2)  # c B H 1 T (8, B, 8, 1, 1)
        x = torch.matmul(x, self.weight)
        x = x.squeeze(-2).permute((3, 1, 2, 0)).reshape(T, B, C)
        return self.out_proj(x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class CrossSelfAttnBlock(nn.Module):
    def __init__(self, n_pos, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", **kwargs):
        super(CrossSelfAttnBlock, self).__init__()
        self.self_attn = StaticSelfAttn(n_pos, d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, cross_res=True, return_attn=False):
        """
        This module is different from transformer decoder layer in the below aspects:
        1. Do cross attention first, then do self attention
        2. Self attention is implemented with static self attention.
        3. Cross attention don't have residue connection when this module is used as the first layer.
        """
        attn_maps = {}
        tgt2, attn_map = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
#         print(tgt2.shape)
        if return_attn:
            attn_maps['cross_attn'] = attn_map.detach().cpu().numpy()
        if cross_res:
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_maps

class StyleBankExtractor(nn.Module):
    def __init__(self, dim_in1, dim_in2, n_layers, n_emb, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(StyleBankExtractor, self).__init__()
        # self.prototype = nn.Parameter(torch.randn(n_emb, d_model)) #512, 64
        self.proj_in1 = nn.Linear(dim_in1, d_model)
        self.proj_in2 = nn.Linear(dim_in2, d_model)
        self.layers = nn.ModuleList([CrossSelfAttnBlock(n_emb, d_model, nhead, dim_feedforward, dropout, activation) for _ in range(n_layers)])

    def forward(self, memory, gcnbedding, return_attn=False, padding_mask=None):
        """
        :param memory: T x B x C
        :return: T x B x C
        """
        _, B, C = memory.shape # 1, bs, attsize(300)
        m = self.proj_in1(memory) # 1, bs, 64
        # output = self.prototype.unsqueeze(1).repeat((1, B, 1))

        output = gcnbedding # B, C(300)
        # output = output.unsqueeze(0)
        output = self.proj_in2(output)

        attn_maps = {}
        for idx, mod in enumerate(self.layers):
            # Todo: Whether the first cross attention should use residue connection
            if idx == 0:
                cross_res = False
            else:
                cross_res = True
            output, attn_map = mod(output, m, cross_res=cross_res, return_attn=return_attn, memory_key_padding_mask=padding_mask)
            attn_maps[idx] = attn_map
            
        return F.normalize(output, dim=-1), attn_maps
    
class OrthoStyleBankExtractor(nn.Module):
    # use ortho style prototype
    def __init__(self, dim_in, n_layers, n_emb, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(OrthoStyleBankExtractor, self).__init__()
        
#         self.prototype = nn.Parameter(torch.randn(n_emb, d_model))
        assert n_emb == d_model, 'In/out dims must be equal for ortho'
        self.log_mat_half = nn.Parameter(torch.randn([n_emb, d_model]))
        
        self.proj_in = nn.Linear(dim_in, d_model)
        self.layers = nn.ModuleList([CrossSelfAttnBlock(n_emb, d_model, nhead, dim_feedforward, dropout, activation) for _ in range(n_layers)])

    def forward(self, memory, return_attn=False, padding_mask=None):
        """
        :param memory: T x B x C
        :return: T x B x C
        """
        _, B, _ = memory.shape
        m = self.proj_in(memory)
        
        prototype = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
        output = prototype.unsqueeze(1).repeat((1, B, 1))
                                         
        attn_maps = {}
        for idx, mod in enumerate(self.layers):
            # Todo: Whether the first cross attention should use residue connection
            if idx == 0:
                cross_res = False
            else:
                cross_res = True
            output, attn_map = mod(output, m, cross_res=cross_res, return_attn=return_attn, memory_key_padding_mask=padding_mask)
            attn_maps[idx] = attn_map
            
        return F.normalize(output, dim=-1), attn_maps
    
class TransformerDecoderLayer_QKV(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", mode='self_attn', kernel_size=1):
        super(TransformerDecoderLayer_QKV, self).__init__()
        self.mode = mode
#         if mode == 'dsconv':
#             self.self_attn = DSConv(1, d_model, d_model, kernel_size, activation=activation)
        if mode == 'self_attn':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
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
        Difference from TransformerDecoderLayer:
        1. cross attention can use additional prototype as key.
        2. self attention can be replaced by ds-conv
        """
        attn_maps = {}

        if self.mode == 'self_attn':
            tgt2, attn_map = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
            if return_attn:
                attn_maps['self_attn'] = attn_map.detach().cpu().numpy()
        else:
            raise NotImplementedError
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
        tgt = self.norm3(tgt)
        
        return tgt, attn_maps
    
class PrototypicalStyleDecoder(nn.Module):
    def __init__(self, n_layers, n_emb, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PrototypicalStyleDecoder, self).__init__()
        self.prototype = nn.Parameter(torch.randn(n_emb, d_model), requires_grad=True)
        # self.positional_encoding = PositionalEmbedding(d_model)
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(TransformerDecoderLayer_QKV(d_model, nhead, dim_feedforward, dropout, activation))

    def forward(self, x, spk_bank, return_attn=False, padding_mask=None):
        """
        :param x: T x B x C
        :param spk_bank: T x B x C
        :return: T x B x C
        """
        _, B, _ = x.shape
        prototype = self.prototype.unsqueeze(1).repeat((1, B, 1))
        # output = x + self.positional_encoding(x, mode='TBC')
        # do not add positional embedding to avoid trivial solution of copy-paste
        output = x
        attn_maps = {}
        for idx, m in enumerate(self.layers):
            output, attn_map = m(output, prototype, spk_bank, return_attn=return_attn, tgt_key_padding_mask=padding_mask)
            attn_maps[idx] = attn_map
        return output, attn_maps