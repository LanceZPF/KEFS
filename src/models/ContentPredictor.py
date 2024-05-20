from src.models.models import ResnetBlock
from src.vq.quantize import VectorQuantizer
from src.vq.gumbel_vector_quantizer import GumbelVectorQuantizer

import torch
import torch.nn as nn

class ContentPredictor(nn.Module):
    def __init__(self, z_channels, embed_dim, n_embed, vq_type = "vq", enable_vq = True, hard = False, exclude_bg = False, vq_groups = 1):
        super(ContentPredictor, self).__init__()
        
        # self.quant_conv = nn.Sequential(
        #                    ResnetBlock(in_channels=z_channels,
        #                                 out_channels=embed_dim,
        #                                 temb_channels=0,
        #                                 dropout=0.0),
        #                    ResnetBlock(in_channels=embed_dim,
        #                                 out_channels=embed_dim,
        #                                 temb_channels=0,
        #                                 dropout=0.0)
        #                   )
        self.quant_conv = nn.Linear(z_channels, embed_dim)
        
        self.exclude_bg = exclude_bg
        self.enable_vq = enable_vq
        self.vq_type = vq_type # vq or gumbel_vq
        if self.enable_vq:
            print(f"Disentangle head using VQ")
            
            if self.vq_type == "vq":
                self.quantizer = VectorQuantizer(n_embed, embed_dim, beta=0.25)
            elif self.vq_type == "gumbel_vq":
                vq_temp = (2,0.1,0.99995)
                vq_groups = vq_groups
                self.quantizer = GumbelVectorQuantizer(dim=embed_dim,
                                                       num_vars=n_embed,
                                                       temp=vq_temp,
                                                       groups=vq_groups,
                                                       combine_groups=False,
                                                       vq_dim=embed_dim,
                                                       time_first=False,
                                                       num_codebook=1,
                                                       other_codebook_sg=False,
                                                       hard = hard
                                                      )
            else:
                raise NotImplementedError("VQ type should be vq or gumbel_vq.")
                
                
                
    def quantize(self, x):

        # x: 416 (B*xxx), 300 (c), 1(h*w)

        # input a feature: B, C, H, W
        # output: quant: B, C, H, W | loss |
        if not self.enable_vq:
            raise NotImplementedError("VQ is not enabled.")

        quant, emb_loss, info = self.quantizer(x)
        # mask = info[2].reshape(B, H, W)

        return quant, emb_loss # actually no soft mask for it!

    def forward(self, x, visual=False):
        # always output the mask

        x = x.squeeze(2)
        
        h = self.quant_conv(x)

        h = h.unsqueeze(-1)

        quant, emb_loss= self.quantize(h)
        
#         if not visual:
#             return quant, emb_loss, None
#         else:
#             B, _, H, W = x.shape
#             mask = info[2].reshape(B, H, W)
        
        return quant, emb_loss