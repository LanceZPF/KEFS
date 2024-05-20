# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        num_codebook=1,
        other_codebook_sg=False,
        hard = False # whether to used hard = Ture or False
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first
        self.hard = hard

        self.num_codebook = num_codebook
        self.other_codebook_sg = other_codebook_sg
        print(f"Gumble-Softmax VQ: other_codebook_sg={self.other_codebook_sg}")

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.ParameterList()
#         self.vars = nn.ModuleList()
        for i in range(self.num_codebook):
            self.vars.append(nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim)))
            nn.init.uniform_(self.vars[-1])

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        assert len(temp) == 3, temp

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None
        
        # for CE loss
        switch_th_rate = (1.0, 1.0, 1.0)
        
        self.switch_th = np.log(num_vars)
        self.switch_th_max, self.switch_th_min, self.switch_th_decay = switch_th_rate
        self.curr_switch_th = self.switch_th * self.switch_th_max
        # self.poisson = poisson

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )
    def get_codebook_indices(self, codebook_idx=0):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars[codebook_idx].device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self, codebook_idx=0):
        indices = self.get_codebook_indices(codebook_idx)
        return (
            self.vars[codebook_idx].squeeze(0)
                .index_select(0, indices)
                .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n, codebook_idx=0):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
                n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars[codebook_idx].squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res
    
#     def detect_code_switch(self, x):
#         # x: logsoftmax-logits, [B, H, W, G, C]
#         # x_argmax is not sampled, but the predicted logits.
#         B, T, G, C = x.shape
#         H = W = int(np.sqrt(T))
#         x = x.reshape(B, H, W, G, C)
        
#         x_argmax = torch.argmax(x, dim=-1)
#         x_ce_h = (F.cross_entropy(F.log_softmax(x[:, :-1, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, 1:, :, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, 1:, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :-1, :, 0], reduction='none')) / 2
#         x_ce_v = (F.cross_entropy(F.log_softmax(x[:, :, :-1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, 1:, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, :, :1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, -1:, 0], reduction='none')) / 2

#         x_ce = torch.cat([x_ce_h.flatten(1), x_ce_v.flatten(1)], -1)
#         switch_point = x_ce > self.curr_switch_th
#         switch_penalty = F.tanh(x_ce / self.curr_switch_th) * self.curr_switch_th

#         return switch_penalty
    
#     def detect_code_switch(self, x):
#         # x: logsoftmax-logits, [B, H, W, G, C]
#         # x_argmax is not sampled, but the predicted logits.
#         B, T, G, C = x.shape
#         H = W = int(np.sqrt(T))
#         x = x.reshape(B, H, W, G, C)
        
#         x_argmax = torch.argmax(x, dim=-1)
        
#         backgroup_mask = (x_argmax == 0)
#         mask_h = ~((backgroup_mask[:, :-1, :, 0] + backgroup_mask[:, 1:, :, 0])>0)
#         mask_v = ~((backgroup_mask[:, :, :-1, 0] + backgroup_mask[:, :, 1:, 0])>0)

#         x_ce_h = mask_h*(F.cross_entropy(F.log_softmax(x[:, :-1, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, 1:, :, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, 1:, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :-1, :, 0], reduction='none')) / 2
#         x_ce_v = mask_v*(F.cross_entropy(F.log_softmax(x[:, :, :-1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, 1:, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, :, :1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, -1:, 0], reduction='none')) / 2

#         x_ce = torch.cat([x_ce_h.flatten(1), x_ce_v.flatten(1)], -1)
#         switch_point = x_ce > self.curr_switch_th
#         switch_penalty = F.tanh(x_ce / self.curr_switch_th) * self.curr_switch_th

#         return switch_penalty
    
    def detect_code_switch(self, x, img_size = None):
        # x: logsoftmax-logits, [B, H, W, G, C]
        # x_argmax is not sampled, but the predicted logits.
        B, T, G, C = x.shape
        
#         if img_size is None:
#             H = W = int(np.sqrt(T))
#         else:
#             H = img_size[0]
#             W = img_size[1]
        if T == 1024:
            H = W = int(np.sqrt(T))
        elif T == 1152:
            H = 48
            W = 24
        else:
            H = W = int(np.sqrt(T))

        x = x.reshape(B, H, W, G, C)
        x_argmax = torch.argmax(x, dim=-1)
        
        # backgroup_mask = (x_argmax == 0)
        # mask_u = ~((backgroup_mask[:, 1:, :-1, 0] + backgroup_mask[:, -1:, 1:, 0])>0)
        # mask_d = ~((backgroup_mask[:, :-1, :-1, 0] + backgroup_mask[:, 1:, 1:, 0])>0)

#         x_ce_h = (F.cross_entropy(F.log_softmax(x[:, :-1, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, 1:, :, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, 1:, :, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :-1, :, 0], reduction='none')) / 2
        
#         x_ce_v = (F.cross_entropy(F.log_softmax(x[:, :, :-1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, 1:, 0], reduction='none') + \
#                 F.cross_entropy(F.log_softmax(x[:, :, :1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :, -1:, 0], reduction='none')) / 2
        
        x_ce_u = (F.cross_entropy(F.log_softmax(x[:, 1:, :-1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :-1, 1:, 0], reduction='none') + \
                F.cross_entropy(F.log_softmax(x[:, :-1, 1:, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, 1:, :-1, 0], reduction='none')) / 2
        x_ce_d = (F.cross_entropy(F.log_softmax(x[:, 1:, 1:, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, :-1, :-1, 0], reduction='none') + \
                F.cross_entropy(F.log_softmax(x[:, :-1, :-1, 0], dim=-1).permute(0, 3, 1, 2), x_argmax[:, 1:, 1:, 0], reduction='none')) / 2
        
#         x_ce = torch.cat([x_ce_h.flatten(1), x_ce_v.flatten(1), x_ce_u.flatten(1), x_ce_d.flatten(1)], -1)
        x_ce = torch.cat([x_ce_u.flatten(1), x_ce_d.flatten(1)], -1)
        switch_point = x_ce > self.curr_switch_th
        switch_penalty = F.tanh(x_ce / self.curr_switch_th) * self.curr_switch_th

        return switch_penalty

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False, padding_mask=None):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)
        
        result['logits'] = x.view(bsz, tsz, self.groups, -1)[:, :, 0, ...]

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if padding_mask is not None:
            avg_probs = torch.softmax(x.reshape(bsz, tsz, self.groups, -1).float(), dim=-1)
            new_avg_probs = torch.zeros_like(avg_probs)
            new_avg_probs[~padding_mask] = avg_probs[~padding_mask]
            avg_probs = new_avg_probs.reshape(bsz * tsz, self.groups, -1)
            avg_probs = avg_probs.sum(dim=0) / (~padding_mask).sum()
            
            avg_probs_bg = avg_probs[:, 1::] # except the bg
            avg_probs_bg = avg_probs_bg / avg_probs_bg.sum(dim=1)
        else:
            log_probs = x.reshape(bsz, tsz, self.groups, -1).float()
            soft_probs = torch.softmax(log_probs, dim=-1)
            kl_chain = F.kl_div(log_probs[:, :-1], soft_probs[:, 1:])
            avg_probs = soft_probs.reshape(bsz*tsz, self.groups, -1).mean(dim=0)
            
            avg_probs_bg = avg_probs[:, 1::] # except the bg
#             print("avg_probs_bg", avg_probs_bg.shape)
            avg_probs_bg = avg_probs_bg / avg_probs_bg.sum(dim=1).unsqueeze(1)
#             print("avg_probs_bg", avg_probs_bg.shape)
            # avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
            
            
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()
        
        result["prob_perplexity_bg"] = torch.exp(
            -torch.sum(avg_probs_bg * torch.log(avg_probs_bg + 1e-7), dim=-1)
        ).sum()
        
        result['kl_chain'] = kl_chain
        
        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)
#             x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x
            # x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        
        ce_chain = torch.mean(self.detect_code_switch(log_probs))
        result['ce_chain'] = ce_chain
        
        soft_mask = x.clone().view(bsz * tsz, self.groups, -1)[:, 0, :]
        result["soft_mask"] = soft_mask
        
        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = [var.repeat(1, self.groups, 1) for var in vars]

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )
        
        res = []
        for idx, var in enumerate(vars):
            if idx > 0 and self.other_codebook_sg:
                r = torch.matmul(x.detach().reshape(bsz * tsz, self.groups, -1).transpose(0, 1),
                                 var[0].reshape(self.groups, -1, var.shape[2]))
            else:
                r = torch.matmul(x.reshape(bsz * tsz, self.groups, -1).transpose(0, 1), var[0].reshape(self.groups, -1, var.shape[2]))
            r = r.transpose(0, 1).contiguous()
            r = r.view(bsz, tsz, -1)

            # x = x.unsqueeze(-1) * vars
            # print(x.shape)
            # x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
            # x = x.sum(-2)
            # print(x.shape)
            # x = x.view(bsz, tsz, -1)
            # print(torch.max(torch.abs(x - x1)))

            if padding_mask is not None:
                r[padding_mask] = 0
            if not self.time_first:
                r = r.transpose(1, 2)  # BTC -> BCT
            
            res.append(r)

        result["x"] = res[0]
        result["xs"] = res
        return result


def gumble_vq_wrapper(module,
                      dim,
                      num_vars,
                      temp,
                      groups,
                      combine_groups,
                      vq_dim,
                      time_first,
                      activation=nn.GELU(),
                      weight_proj_depth=1,
                      weight_proj_factor=1,
                      vq_pos=-1):
    
    class GumbleVQModel(nn.Module):
        def __init__(self):
            super(GumbleVQModel, self).__init__()
            self.module = module
            self.vq_pos = vq_pos
            self.quantizer = GumbelVectorQuantizer(dim,
                                                   num_vars,
                                                   temp,
                                                   groups,
                                                   combine_groups,
                                                   vq_dim,
                                                   time_first,
                                                   activation,
                                                   weight_proj_depth,
                                                   weight_proj_factor)

        def set_num_updates(self, num_updates):
            self.quantizer.set_num_updates(num_updates)

        def forward(self, x):
            results = self.module(x)
            if isinstance(results, tuple):
                vq_results = self.quantizer(results[self.vq_pos])
                results = list(results)
                results.append(vq_results)
                return results
            else:
                vq_results = self.quantizer(results)
                return results, vq_results
    return GumbleVQModel()
