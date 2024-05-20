import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
import torch
from torch.nn import *
import pickle

from mmdet.models.graphutil import *

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 semantic_dim=300,
                 adj_file1=None,
                 adj_file2=None,
                 adj_file3=None,
                 inp_name=None,
                 lambda1=0,
                 lambda2=0,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.semantic_dim = semantic_dim

        self.num_classes = num_classes
        self.attr_dim = semantic_dim

        self.t = 0.4
        #TODO: t into config

        self.adj_file1 = adj_file1
        self.adj_file2 = adj_file2
        self.adj_file3 = adj_file3
        self.inp_name = inp_name

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)

            self.lrelu = nn.LeakyReLU(0.2)
            _adj1 = gen_A(self.num_classes, self.t, self.adj_file1)
            self.A1 = Parameter(_adj1.float()).cuda()
            _adj2 = gen_A(self.num_classes, self.t, self.adj_file2)
            self.A2 = Parameter(_adj2.float()).cuda()
            _adj3 = gen_A(self.num_classes, self.t, self.adj_file3)
            self.A3 = Parameter(_adj3.float()).cuda()

            self.inp = torch.Tensor(np.load(self.inp_name)).cuda()
            
            self.gc11 = GraphConvolution(self.attr_dim, 1024, bias=True)
            self.gc21 = GraphConvolution(1024, self.num_classes, bias=True)

            # self.gc12 = GraphConvolution(self.attr_dim, 1024, bias=True)
            # self.gc22 = GraphConvolution(1024, self.cls_last_dim, bias=True)
            #
            # self.gc13 = GraphConvolution(self.attr_dim, 1024, bias=True)
            # self.gc23 = GraphConvolution(1024, self.cls_last_dim, bias=True)

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_cls_feats=False):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.with_cls:

            cls_score = self.fc_cls(x_cls)

            if self.gzsd:
                y_seen_bg = torch.mm(cls_score, self.seen_bg_weight[:, None]) + self.seen_bg_bias
                cls_score = torch.cat((cls_score, y_seen_bg), dim=1)

            inp = self.inp

            adj1 = gen_adj(self.A1).detach()
            cls_embed = self.gc11(inp, adj1)
            cls_embed = self.lrelu(cls_embed)
            cls_embed = self.gc21(cls_embed, adj1)

            # adj2 = gen_adj(self.A2).detach()
            # cls_score2 = self.gc12(inp, adj2)
            # cls_score2 = self.lrelu(cls_score2)
            # cls_score2 = self.gc22(cls_score2, adj2)
            #
            # adj3 = gen_adj(self.A3).detach()
            # cls_score3 = self.gc13(inp, adj3)
            # cls_score3 = self.lrelu(cls_score3)
            # cls_score3 = self.gc23(cls_score3, adj3)
            #
            # cls_score = self.lambda2 * (self.lambda1*cls_score1 + (1-self.lambda1)*cls_score2) \
            #     + (1-self.lambda2) * cls_score3

            cls_embed = cls_embed.squeeze().transpose(0, 1)

            cls_score = torch.matmul(cls_score, cls_embed)

        else:
            cls_score = None
        #cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        if return_cls_feats:
            return cls_score, bbox_pred, x_cls
        else:
            return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, num_classes=0, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            num_classes=num_classes,
            *args,
            **kwargs)
