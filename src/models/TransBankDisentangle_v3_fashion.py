from src.models.models import Encoder, Decoder, ResnetBlock
from src.models.TransEncoder import TransEncoder, TransDecoder
from src.models.ContentPredictor import ContentPredictor
from src.models.transformer import StyleBankExtractor, PrototypicalStyleDecoder
from src.loss.centor_loss import concentration_loss
from src.loss.background_loss import BGLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransBankDisentangle(nn.Module):
    def __init__(self, net_args):
        super(TransBankDisentangle, self).__init__()
        
        self.net_args = net_args
        
        # auto-encoder: using transformer encoder!
        self.encoder = TransEncoder(img_size = 128, in_chans = net_args['in_channels'], 
                                    out_chans = net_args['z_channels'], embed_dim = 192, depth = 6, num_heads = 4)
        
        self.decoder = TransDecoder(img_size = 128, in_chans = net_args['z_channels'], 
                                    out_chans = net_args['in_channels'], embed_dim = 192, depth = 6, num_heads = 4, 
                                    use_pos_embed = net_args['use_pos_embed'])

        # vq
        self.content_extractor = ContentPredictor(net_args['z_channels'], net_args['embed_dim'], net_args['n_embed'], vq_type = net_args['vq_type'], hard = net_args['hard'])

        # transformer part
        self.style_bank = StyleBankExtractor(dim_in = net_args['dim_in'], n_layers = net_args['n_layers'], 
                                            n_emb = net_args['n_emb_style'], d_model = net_args['d_model'], 
                                            nhead = net_args['nhead'], dim_feedforward = net_args['dim_feedforward'])

        self.style_mixer = PrototypicalStyleDecoder(net_args['n_layers'], net_args['n_emb_style'], net_args['d_model'], 
                                               net_args['nhead'], net_args['dim_feedforward'])
        
        self.bg_loss = BGLoss()

    def forward(self, x, padding_mask=None, mode='train'):
        """
        :param x: B x C x H x W
        :param mode:
        :return: 
        """
        if mode == 'train':
            h = self.encoder(x)
            content, emb_loss, mask, soft_mask = self.content_extractor(h) # B, C, H', W'

            B, C, _H, _W = content.shape
            h = h.reshape(B, C, _H * _W).permute(2, 0, 1) # B, C, H', W' -> H'*W', B, C
            content = content.reshape(B, C, _H * _W).permute(2, 0, 1)
            
            # here perform the center loss
            if self.net_args['vq_type'] == "gumbel_vq":
                # soft_mask B * H * W, C
#                 pred_seg_idx = mask.reshape(B*_H*_W, 1)
#                 pred_seg = torch.zeros(pred_seg_idx.shape[0], self.net_args['n_embed']).to(pred_seg_idx)
#                 pred_seg.scatter_(1, pred_seg_idx, 1)
#                 pred_seg = pred_seg.reshape(B, _H, _W, self.net_args['n_embed']).permute(0, 3, 1, 2) # here need B, C, _H, _W
                pred_seg = soft_mask.reshape(B, _H, _W, self.net_args['n_embed']).permute(0, 3, 1, 2) # B, C, _H, _W
                center_loss = concentration_loss(pred_seg)
        
                bg_loss = self.bg_loss(pred_seg)
            else:
                center_loss = 0
                bg_loss = 0

            if self.net_args['shuffle']:
                idx = torch.randperm(h.shape[0])
                h = h[idx].view(h.size())
            
            style, _ = self.style_bank(h)
            remix_feat, _ = self.style_mixer(content, style)

            remix_feat = remix_feat.permute(1, 2, 0).reshape(B, C, _H, _W)

            recon = self.decoder(remix_feat)

            return recon, emb_loss, center_loss, bg_loss
        
        elif mode == 'proj':
#             return self.content_extractor(x)  # B C T
            raise NotImplementedError
        else:
            raise NotImplementedError
            
    def exchange(self, x, y, padding_mask=None, mode='train'):
        """
        exchange the style and content to recon
        :param x: B x C x H x W
        :param mode:
        :return: 
        """
        if mode == 'train':
            
            # extract full feature
            h_x = self.encoder(x)
            h_y = self.encoder(y)
            
            # extract content
            content_x, emb_loss_x, mask_x, soft_mask_x = self.content_extractor(h_x) # B, C, H', W'
            content_y, emb_loss_y, mask_y, soft_mask_y = self.content_extractor(h_y) # B, C, H', W'
            
            emb_loss = emb_loss_x + emb_loss_y
            
            B, C, _H, _W = content_x.shape
            
            h_x = h_x.reshape(B, C, _H * _W).permute(2, 0, 1) # B, C, H', W' -> H'*W', B, C
            content_x = content_x.reshape(B, C, _H * _W).permute(2, 0, 1)
            h_y = h_y.reshape(B, C, _H * _W).permute(2, 0, 1) # B, C, H', W' -> H'*W', B, C
            content_y = content_y.reshape(B, C, _H * _W).permute(2, 0, 1)
            
            # here perform the center loss
            if self.net_args['vq_type'] == "gumbel_vq":
                pred_seg_x = soft_mask_x.reshape(B, _H, _W, self.net_args['n_embed']).permute(0, 3, 1, 2) # B, C, _H, _W
                center_loss_x = concentration_loss(pred_seg_x)
                bg_loss_x = self.bg_loss(pred_seg_x)
            
                pred_seg_y = soft_mask_y.reshape(B, _H, _W, self.net_args['n_embed']).permute(0, 3, 1, 2) # B, C, _H, _W
                center_loss_y = concentration_loss(pred_seg_y)
                bg_loss_y = self.bg_loss(pred_seg_y)
            
                center_loss = center_loss_x + center_loss_y
                bg_loss = bg_loss_x + bg_loss_y
                
            else:
                center_loss = 0
                bg_loss = 0
            
            style_x, _ = self.style_bank(h_x)
            style_y, _ = self.style_bank(h_y)
            
            remix_feat_x, _ = self.style_mixer(content_x, style_y)
            remix_feat_x = remix_feat_x.permute(1, 2, 0).reshape(B, C, _H, _W)
            
            remix_feat_y, _ = self.style_mixer(content_y, style_x)
            remix_feat_y = remix_feat_y.permute(1, 2, 0).reshape(B, C, _H, _W)

            recon_x = self.decoder(remix_feat_x)
            recon_y = self.decoder(remix_feat_y)

            return recon_x, recon_y, emb_loss, center_loss, bg_loss
        
        elif mode == 'proj':
#             return self.content_extractor(x)  # B C T
            raise NotImplementedError
        else:
            raise NotImplementedError