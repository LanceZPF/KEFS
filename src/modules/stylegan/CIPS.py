from src.modules.stylegan.blocks import LFF, StyledConv, ToRGB, convert_to_coord_format

import torch
from torch import nn
import torch.nn.functional as F

class StyleLFF(nn.Module):
    """ A Fourier Embedding MLP for ViTGAN
    window_size: P
    in_features: D
    """
    def __init__(self, window_size, in_features, hidden_features, activation = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        style_dim = in_features
        demodulate = True
        
        self.fc1 = StyledConv(in_features, hidden_features, 1, style_dim, demodulate=demodulate, activation=activation)
        self.fc2 = ToRGB(hidden_features, style_dim, upsample=False)
        
        self.coords = None
        self.lff = LFF(in_features)
        self.window_size = window_size
        
    def forward(self, style):
        """ 
        Efou [B, P^2·D]
        Style [B, D]
        Output [B, P^2.C]
        """ 
        if self.coords is None:
            device = style.device
            self.coords = convert_to_coord_format(style.size(0), self.window_size, 
                                                  self.window_size, device, integer_values=False)
        
        x = self.lff(self.coords)
        
        x = self.fc1(x, style)
        x = self.fc2(x, style)
        
        return x