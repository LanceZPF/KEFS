from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import model
import dmodel
from cls_models import ClsModel, ClsUnseen
import torch.nn.functional as F
import torch.nn as nn
import random
import losses

from torch.nn import Parameter
from mmdet.models.graphutil import *

from src.models.TransEncoder import TransEncoder
from src.models.transformer import StyleBankExtractor
from src.models.StyleTransDecoder_v3 import StyleTransDecoder

from ldm.ldm.util import instantiate_from_config

from omegaconf import OmegaConf

import VAE

class TransBankDisentangle(nn.Module):
    def __init__(self, opt):
        super(TransBankDisentangle, self).__init__()

        self.opt = opt

        self.encoder = TransEncoder(embed_dim=opt.attSize, out_chans=opt.z_channels, depth=6, num_heads=4)

        self.cross_attention = StyleBankExtractor(dim_in1=opt.attSize, dim_in2=opt.attSize, n_layers=opt.n_layers,
                                             n_emb=opt.n_emb_style, d_model=opt.d_model,
                                             nhead=opt.nhead, dim_feedforward=opt.dim_feedforward)

        self.style_bank = StyleBankExtractor(dim_in1=opt.z_channels, dim_in2=opt.z_channels, n_layers=opt.n_layers,
                                             n_emb=opt.n_emb_style, d_model=opt.d_model,
                                             nhead=opt.nhead, dim_feedforward=opt.dim_feedforward)

        self.style_mixer = StyleTransDecoder(in_chans=opt.z_channels,
                                             in_chans_style=opt.d_model, out_chans=opt.attSize,
                                             embed_dim=192,
                                             n_emb_style=opt.n_emb_style, depth=6, num_heads=4)

    def forward(self, g1, g2, g3):

        content = g1

        content = self.encoder(content)

        content = content.permute(2, 0, 1)  # B, C, 1 -> 1, B, C

        g2 = g2.permute(2, 0, 1)
        g3 = g3.permute(2, 0, 1)

        s, _ = self.cross_attention(g2, g3) # 1, B, C

        style, __ = self.style_bank(content, s)

        recon = self.style_mixer(content, style)

        return recon

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

class GraphConvolutionNetwork(nn.Module):

    def __init__(self, opt):
        super(GraphConvolutionNetwork, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)

        self.num_classes = opt.nclass_all
        self.attr_dim = opt.attSize

        self.t = 0.4
        # TODO: t into config

        self.adj_file1 = opt.adj_file1
        self.adj_file2 = opt.adj_file2
        self.adj_file3 = opt.adj_file3
        self.inp_name = opt.inp_name

        self.lambda1 = opt.glambda1
        self.lambda2 = opt.glambda2

        _adj1 = gen_A(self.num_classes, self.t, self.adj_file1)
        self.A1 = Parameter(_adj1.float()).cuda()
        _adj2 = gen_A(self.num_classes, self.t, self.adj_file2)
        self.A2 = Parameter(_adj2.float()).cuda()
        _adj3 = gen_A(self.num_classes, self.t, self.adj_file3)
        self.A3 = Parameter(_adj3.float()).cuda()

        self.cls_last_dim = opt.resSize

        # self.gc11 = GraphConvolution(self.attr_dim, 1024, bias=True)
        # self.gc21 = GraphConvolution(1024, 300, bias=True)

        self.gc11 = GraphConvolution(self.attr_dim, 1024)
        self.gc21 = GraphConvolution(1024, 300)

        self.gc12 = GraphConvolution(self.attr_dim, 1024)
        self.gc22 = GraphConvolution(1024, 300)

        self.gc13 = GraphConvolution(self.attr_dim, 1024)
        self.gc23 = GraphConvolution(1024, 300)

    def forward(self, inp):

        adj1 = gen_adj(self.A1).detach()
        cls_score1 = self.gc11(inp, adj1)
        cls_score1 = self.lrelu(cls_score1)
        cls_score1 = self.gc21(cls_score1, adj1)

        adj2 = gen_adj(self.A2).detach()
        cls_score2 = self.gc12(inp, adj2)
        cls_score2 = self.lrelu(cls_score2)
        cls_score2 = self.gc22(cls_score2, adj2)

        adj3 = gen_adj(self.A3).detach()
        cls_score3 = self.gc13(inp, adj3)
        cls_score3 = self.lrelu(cls_score3)
        cls_score3 = self.gc23(cls_score3, adj3)

        # cls_score = self.lambda2 * (self.lambda1 * cls_score1 + (1 - self.lambda1) * cls_score2) \
        #             + (1 - self.lambda2) * cls_score3

        # cls_score = self.Transcoder(cls_score1.unsqueeze(-1), cls_score2.unsqueeze(-1), cls_score3.unsqueeze(-1))

        cls_score = [cls_score1.squeeze().unsqueeze(-1), cls_score2.squeeze().unsqueeze(-1), cls_score3.squeeze().unsqueeze(-1)]

        return cls_score

class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, writter, seen_feats_mean, gen_type='FG'):
        self.tensorwrite = writter

        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.unseen_classifier = ClsUnseen(unseenAtt)
        self.unseen_classifier.cuda()

        self.unseen_classifier = loadUnseenWeights(opt.pretrain_classifier_unseen, self.unseen_classifier)
        self.classifier = ClsModel(num_classes=opt.nclass_all)
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier)

        for p in self.classifier.parameters():
            p.requires_grad = False

        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        # self.netG1 = model.MLP_G(self.opt)
        self.netG1 = model.Generator(self.opt)
        # self.netD = model.MLP_CRITIC(self.opt)
        self.netD = model.MLP_D(self.opt)
        # MLP_3HL_CRITIC

        self.gcn = GraphConvolutionNetwork(self.opt)
        self.Transcoder = TransBankDisentangle(self.opt)

        self.netG = dmodel.LatentDiffusion(self.opt)

        ##add
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.opt.featnorm = True
        self.inter_contras_criterion = losses.SupConLoss_clear(self.opt.inter_temp)
        ##

        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()
            self.gcn = self.gcn.cuda()
            self.Transcoder = self.Transcoder.cuda()
            self.netG1 = self.netG1.cuda()

        print('\n\n#############################################################\n')
        print(self.netG1, '\n')
        print(self.netD, '\n')
        print(self.gcn, '\n')
        print(self.Transcoder, '\n')
        print(self.netG)
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        self.lm = nn.LogSoftmax(dim=1)

        if self.opt.cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()
            self.cross_entropy_loss.cuda()
            self.inter_contras_criterion.cuda()
            self.lm.cuda()

        self.optimizerG1 = optim.Adam(self.netG1.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr*0.05, betas=(self.opt.beta1, 0.999))
        self.optimizerTranscoder = optim.Adam(self.Transcoder.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerGCN = optim.Adam(self.gcn.parameters(), lr=self.opt.lr*1e3, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        self.output = self.trainEpoch()
        return self.output

    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG1.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        self.netG.load_state_dict(torch.load(self.opt.netDM)['state_dict'])
        self.gcn.load_state_dict(torch.load(self.opt.netGCN)['state_dict'])
        self.Transcoder.load_state_dict(torch.load(self.opt.netT)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch

    ##todo
    def load_pretrain_checkpoint(self):
        checkpoint = torch.load(self.opt.pretrain_GAN_netG)
        self.netG1.load_state_dict(checkpoint['state_dict'])
        self.netD.load_state_dict(torch.load(self.opt.pretrain_GAN_netD)['state_dict'])
        self.netG.load_state_dict(torch.load(self.opt.pretrain_netDM)['state_dict'])
        self.gcn.load_state_dict(torch.load(self.opt.pretrain_netGCN)['state_dict'])
        self.Transcoder.load_state_dict(torch.load(self.opt.pretrain_netT)['state_dict'])
        print(f"loaded weights from best GAN model")

    ##

    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG1.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/dm_{state}.pth')
        torch.save({'state_dict': self.gcn.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gcn_{state}.pth')
        torch.save({'state_dict': self.Transcoder.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/trans_{state}.pth')

    ##todo
    # def save_each_epoch_checkpoint(self, state='latest'):
    #     torch.save({'state_dict': self.netG1.state_dict(), 'epoch': self.epoch},
    #                f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, embedding_gcnv, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features
            2) labels of synthesised  features
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num, self.opt.resSize)
        syn_label = torch.LongTensor(nclass * num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        syn_emb = torch.FloatTensor(num, self.opt.nz)

        embedding_gcnv1, embedding_gcnv2, embedding_gcnv3 = embedding_gcnv

        embedding_gcnv = self.Transcoder(embedding_gcnv1, embedding_gcnv2, embedding_gcnv3)

        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
            syn_emb = syn_emb.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)

                    iclass_emb = embedding_gcnv[i]
                    # iclass_emb = self.Transcoder(iclass_emb1, iclass_emb2, iclass_emb3)

                    syn_emb.copy_(iclass_emb.repeat(num, 1))

                    output = self.netG1(Variable(syn_noise), Variable(syn_att), syn_emb)

                    syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i * num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]

                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)

                iclass_emb = embedding_gcnv[i]
                # iclass_emb = self.Transcoder(iclass_emb1, iclass_emb2, iclass_emb3)

                syn_emb.copy_(iclass_emb.repeat(num, 1))

                output = self.netG1(Variable(syn_noise), Variable(syn_att), syn_emb)

                syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i * num, num).fill_(label)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        if 'BG' == self.gen_type:
            batch_label *= 0
        return batch_feature, batch_label, batch_att, torch.from_numpy(self.attributes)

    def calc_gradient_penalty(self, real_data, fake_data, input_att, embedding_gcn, contra=False):
        if contra:
            alpha = torch.rand(real_data.size(0), 1)
        else:
            alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att), Variable(embedding_gcn))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    #############################
    def get_z_random(self):
        """
        returns normal initialized noise tensor
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def compute_contrastive_loss(self, feat_q, feat_k):
        # feat_q = F.softmax(feat_q, dim=1)
        # feat_k = F.softmax(feat_k, dim=1)
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

    def latent_augmented_sampling(self):
        query = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
        pos = torch.cuda.FloatTensor(query.shape).uniform_(-self.opt.radius, self.opt.radius).add_(query)
        negs = []
        for k in range(self.opt.num_negative):
            neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
            while (neg - query).abs().min() < self.opt.radius:
                neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
            negs.append(neg)
        return query, pos, negs

    def get_z_random_v2(self, batchSize, nz, random_type='gauss'):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z

    def trainEpoch(self):
        for i in range(0, self.ntrain, self.opt.batch_size):
            # import pdb; pdb.set_trace()
            # input_res, input_label, input_att = self.sample()
            input_res, input_label, real_att, att_matrix = self.sample()

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, real_att, att_matrix = input_res.type(torch.FloatTensor).cuda(), input_label.type(
                torch.LongTensor).cuda(), real_att.type(torch.FloatTensor).cuda(), att_matrix.type(torch.FloatTensor).cuda()

            input_att = Variable(real_att)

            att_matrixv = Variable(att_matrix)
            att_embeddings = self.gcn(att_matrixv)

            att_embeddings1, att_embeddings2, att_embeddings3 = att_embeddings

            embedding_gcn1 = att_embeddings1[input_label]
            embedding_gcn2 = att_embeddings2[input_label]
            embedding_gcn3 = att_embeddings3[input_label]

            embedding_gcn = self.Transcoder(embedding_gcn1, embedding_gcn2, embedding_gcn3)

            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG

                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                embedding_gcnv = Variable(embedding_gcn)

                criticD_real = self.netD(input_resv, input_attv, embedding_gcnv)
                criticD_real = criticD_real.mean().unsqueeze(0)
                criticD_real.backward(self.mone)

                ##real inter contra loss
                input_res_norm = F.normalize((input_resv), dim=1)
                real_inter_contras_loss = self.inter_contras_criterion(input_res_norm, input_label)
                real_inter_contras_loss = real_inter_contras_loss.requires_grad_()
                real_inter_contras_loss.backward()

                ##
                z_random = self.get_z_random()
                query, pos, negs = self.latent_augmented_sampling()
                z_random2 = [query, pos] + negs

                z_conc = torch.cat([z_random] + z_random2, 0)
                label_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)

                embedding_gcn_conc = torch.cat([embedding_gcnv] * (self.opt.num_negative + 3), 0)

                fake = self.netG1(z_conc, label_conc, embedding_gcn_conc)
                fake1 = fake[:input_resv.size(0)]
                fake2 = fake[input_resv.size(0):]
                ##

                criticD_fake = self.netD(fake1.detach(), input_attv, embedding_gcnv)
                criticD_fake = criticD_fake.mean().unsqueeze(0)
                criticD_fake.backward(self.one)

                # gradient penalty
                ##todo
                # gradient_penalty = self.calc_gradient_penalty(input_res, fake1.data, input_att)
                gradient_penalty = self.calc_gradient_penalty(input_res, fake1.data, input_att, embedding_gcnv, contra=False)
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # D_cost.backward()

                criticD_real2 = self.netD(input_resv, input_attv, embedding_gcnv)
                criticD_real2 = criticD_real2.mean().unsqueeze(0)
                criticD_real2.backward(self.mone)

                criticD_fake2 = self.netD(fake2.detach(), input_attv.repeat(self.opt.num_negative + 2, 1), embedding_gcnv.repeat(self.opt.num_negative + 2, 1))
                ##todo
                # criticD_fake2 = criticD_fake2[:input_resv.size(0)]
                ##
                criticD_fake2 = criticD_fake2.mean().unsqueeze(0)
                criticD_fake2.backward(self.one)

                # gradient penalty
                ##todo
                # gradient_penalty2 = self.calc_gradient_penalty(input_res, fake2.data[:input_resv.size(0)], input_att)
                gradient_penalty2 = self.calc_gradient_penalty(input_res.repeat(self.opt.num_negative + 2, 1),
                                                               fake2.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),
                                                               embedding_gcn.repeat(self.opt.num_negative + 2, 1),
                                                               contra=True)
                ##
                gradient_penalty2.backward()

                Wasserstein_D2 = criticD_real2 - criticD_fake2
                D_cost2 = criticD_fake2 - criticD_real2 + gradient_penalty2
                # D_cost2.backward()

                self.optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            self.netG1.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            ##
            z_random = self.get_z_random()
            query, pos, negs = self.latent_augmented_sampling()
            z_random2 = [query, pos] + negs

            z_conc = torch.cat([z_random] + z_random2, 0)
            label_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)
            embedding_gcn_conc = torch.cat([embedding_gcnv] * (self.opt.num_negative + 3), 0)

            fake = self.netG1(z_conc, label_conc, embedding_gcn_conc)
            fake1 = fake[:input_resv.size(0)]
            fake2 = fake[input_resv.size(0):]
            ##

            criticG_fake = self.netD(fake1, input_attv, embedding_gcnv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake

            ##todo
            criticG_fake2 = self.netD(fake2[:input_resv.size(0)], input_attv, embedding_gcnv)
            # criticG_fake2 = self.netD(fake2, input_attv.repeat(self.opt.num_negative+2, 1))
            ##
            criticG_fake2 = criticG_fake2.mean()
            G_cost2 = criticG_fake2

            ##inter contra loss
            input_res_norm_2 = F.normalize((input_resv), dim=1)
            fake_res1 = F.normalize((fake1), dim=1)
            fake_res2 = F.normalize((fake2[:input_resv.size(0)]), dim=1)

            all_features = torch.cat((fake_res1, fake_res2, input_res_norm_2.detach()), dim=0)
            fake_inter_contras_loss = self.inter_contras_criterion(all_features,
                                                                   torch.cat((input_label, input_label, input_label),
                                                                             dim=0))
            # fake_inter_contras_loss.requires_grad_()
            fake_inter_contras_loss = self.opt.inter_weight * fake_inter_contras_loss
            # fake_inter_contras_loss.backward(retain_graph=True)
            #####################################################################

            self.loss_contra = 0.0
            for j in range(input_res.size(0)):
                logits = fake2[j:fake2.shape[0]:input_res.size(0)].view(self.opt.num_negative + 2, -1)

                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)

                self.loss_contra += self.compute_contrastive_loss(logits[0:1], logits[1:])

            loss_lz = self.opt.lambda_contra * self.loss_contra

            # ---------------------
            # classification loss
            ##
            c_errG = self.cls_criterion(self.classifier(feats=fake1, classifier_only=True), Variable(input_label))

            c_errG = self.opt.cls_weight * c_errG
            # --------------------------------------------

            gcn_loss = self.cls_criterion(self.lm(embedding_gcnv), Variable(input_label))

            errG = - G_cost - G_cost2 + c_errG + loss_lz + fake_inter_contras_loss + gcn_loss

            errG.backward()

            self.optimizerG1.step()

            self.optimizerGCN.step()

            self.optimizerTranscoder.step()

            #TODO: GCN denoising classification loss
            #
            # GCN loss: {gcn_loss.data.item(): .4f}
            print(f"{self.gen_type} [{self.epoch + 1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
                        Loss: {errG.item() :0.4f} D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f} GCN loss: {gcn_loss.data.item():.4f} \
                        W dist: {Wasserstein_D.data.item():.4f} seen loss: {c_errG.data.item():.4f} loss div: {loss_lz.item():0.4f} \
                        real_inter_contras_loss: {real_inter_contras_loss.data.item():.4f} fake_inter_contras_loss : {fake_inter_contras_loss.data.item():.4f}")


            input_resv = Variable(input_res.unsqueeze(-2))
            input_attv = Variable(fake1.unsqueeze(-2).detach())

            loss_dm, loss_dic = self.netG(input_resv, input_attv)

            loss_dm.backward()
            self.optimizerG.step()

            print(loss_dic)

            self.tensorwrite.add_scalar("train_loss_GAN", errG.item(), self.epoch * self.ntrain + i)
            self.tensorwrite.add_scalar("train_loss_df", loss_dm, self.epoch*self.ntrain + i)

        self.netG1.eval()
        self.netG.eval()
        return input_att, att_embeddings
