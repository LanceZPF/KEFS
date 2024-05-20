import torch
import torch.nn.functional as F

def orthonomal_loss(w):
    B, K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.bmm(w_norm, w_norm.permute(0, 2, 1))

    return F.mse_loss(WWT - torch.eye(K).unsqueeze(0).cuda(), torch.zeros(B, K, K).cuda())

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 2) + epsilon, 0.5).unsqueeze(2).expand_as(feature)
    return torch.div(feature, norm)