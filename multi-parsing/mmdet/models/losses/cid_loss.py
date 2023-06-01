import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4): #超参数，用于调整正负样本的权重
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask=None): #pred是模型预测的概率，gt是对应的真实标签，mask是可选的用于控制样本的损失计算
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
              pred (batch x c x h x w)
              gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float() #所有gt等于1的位置，将对应的pos_inds设置为1，否则将对应的neg_inds设置为1。
        neg_inds = gt.lt(1).float()

        if mask is not None:
            pos_inds = pos_inds * mask
            neg_inds = neg_inds * mask

        neg_weights = torch.pow(1 - gt, self.beta) #使用beta参数计算负样本的权重

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05): #计算对比损失的标量温度
        super(ContrastiveLoss, self).__init__()
        self.temp = temperature

    def forward(self, features):
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp #计算每个样本和所有其他样本的相似度
        targets = torch.arange(n, dtype=torch.long).cuda()

        loss = F.cross_entropy(logits, targets, reduction='sum') #计算相似度矩阵的交叉熵损失，使用从0到n-1的整数作为类别标签
        return loss