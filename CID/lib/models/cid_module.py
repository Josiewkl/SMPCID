import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math

from .loss import FocalLoss, ContrastiveLoss

def build_iia_module(cfg):
    return IIA(cfg)

def build_gfd_module(cfg):
    return GFD(cfg)

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class IIA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.IIA.IN_CHANNELS
        self.out_channels = cfg.MODEL.IIA.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints + 1 #判断模型输出的通道数是否等于关键点的数量加上一个背景通道
        self.prior_prob = cfg.MODEL.BIAS_PROB

        self.keypoint_center_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0) #计算关键点中心热力图，1x1 的卷积核，步长为 1，填充为 0
        torch.nn.init.normal_(self.keypoint_center_conv.weight, std=0.001) #初始化卷积层的权重，使用正态分布，标准差为 0.001
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob) #计算卷积层的偏置值
        torch.nn.init.constant_(self.keypoint_center_conv.bias, bias_value) #初始化卷积层的偏置，使用常数初始化，值为 bias_value

        self.heatmap_loss = FocalLoss()
        self.contrastive_loss = ContrastiveLoss() #定义对比损失函数

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST #在测试时是否对输入图像进行翻转
        self.max_proposals = cfg.TEST.MAX_PROPOSALS #每张图像最多检测的物体数量
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD #关键点的置信度阈值
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL #中心池化核大小
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1 #池化阈值1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2 #池化阈值2

    def forward(self, features, batch_inputs=None):
        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features)) #iia预测值：sigmoid函数放缩到0~1之间

        if self.training: #如果模型处于训练模式，则使用batch_inputs对象执行训练；否则，返回一个instances对象，该对象包含模型对输入的预测
            gt_multi_heatmap = [x['multi_heatmap'].unsqueeze(0).to(self.device) for x in batch_inputs]
            gt_multi_heatmap = torch.cat(gt_multi_heatmap, dim=0)
            gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(self.device) for x in batch_inputs]
            gt_multi_mask = torch.cat(gt_multi_mask, dim=0) #获取多个热图的真实标签和掩码

            multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap, gt_multi_heatmap, gt_multi_mask) #计算出多个热图之间的损失，结果对比

            contrastive_loss = 0
            total_instances = 0
            instances = defaultdict(list)
            for i in range(features.size(0)): #对每个实例计算对比损失contrastive_loss
                if 'instance_coord' not in batch_inputs[i]: continue
                instance_coord = batch_inputs[i]['instance_coord'].to(self.device)
                instance_heatmap = batch_inputs[i]['instance_heatmap'].to(self.device)
                instance_mask = batch_inputs[i]['instance_mask'].to(self.device)
                instance_imgid = i * torch.ones(instance_coord.size(0), dtype=torch.long).to(self.device)
                instance_param = self._sample_feats(features[i], instance_coord)
                contrastive_loss += self.contrastive_loss(instance_param)
                total_instances += instance_coord.size(0)

                instances['instance_coord'].append(instance_coord)
                instances['instance_imgid'].append(instance_imgid)
                instances['instance_param'].append(instance_param)
                instances['instance_heatmap'].append(instance_heatmap)
                instances['instance_mask'].append(instance_mask)
            
            for k, v in instances.items():
                instances[k] = torch.cat(v, dim=0)

            return multi_heatmap_loss, contrastive_loss/total_instances, instances
        else: #使用预测的热图来推断关键点的位置
            instances = {}
            W = pred_multi_heatmap.size()[-1]
            if self.flip_test:
                center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(dim=0, keepdim=True)
            else:
                center_heatmap = pred_multi_heatmap[:, -1, :, :]

            center_pool = F.avg_pool2d(center_heatmap, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)
            center_heatmap = (center_heatmap + center_pool) / 2.0
            maxm = self.hierarchical_pool(center_heatmap)
            maxm = torch.eq(maxm, center_heatmap).float()
            center_heatmap = center_heatmap * maxm
            scores = center_heatmap.view(-1)
            scores, pos_ind = scores.topk(self.max_proposals, dim=0)
            select_ind = (scores > (self.keypoint_thre)).nonzero()
            if len(select_ind) > 0:
                scores = scores[select_ind].squeeze(1)
                pos_ind = pos_ind[select_ind].squeeze(1)
                x = pos_ind % W
                y = (pos_ind / W).long()
                instance_coord = torch.stack((y, x), dim=1)
                instance_param = self._sample_feats(features[0], instance_coord)
                instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device)
                if self.flip_test:
                    instance_param_flip = self._sample_feats(features[1], instance_coord)
                    instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                    instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
                    instance_param = torch.cat((instance_param, instance_param_flip), dim=0)
                    instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)

                instances['instance_coord'] = instance_coord
                instances['instance_imgid'] = instance_imgid
                instances['instance_param'] = instance_param
                instances['instance_score'] = scores

            return instances
    
    def _sample_feats(self, features, pos_ind): #pos_ind是一个二维数组，表示特征图中需要提取特征的位置
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0) #将维度从 (channels, batch_size) 转换为 (batch_size, channels)

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0 #计算输入张量的高度和宽度维度的平均值
        if map_size > self.pool_thre1:
            maxm = F.max_pool2d(heatmap, 7, 1, 3) #执行最大池化，内核大小为 7，步幅为 3
        elif map_size > self.pool_thre2:
            maxm = F.max_pool2d(heatmap, 5, 1, 2) #执行最大池化时内核大小为 5，步幅为 2
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1) #执行最大池化，内核大小为 3，步幅为 1
        return maxm

class GFD(nn.Module): #人体关键点检测
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.GFD.IN_CHANNELS
        self.channels = cfg.MODEL.GFD.CHANNELS
        self.out_channels = cfg.MODEL.GFD.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints
        self.prior_prob = cfg.MODEL.BIAS_PROB

        self.conv_down = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.c_attn = ChannelAtten(self.in_channels, self.channels) #通道注意力层
        self.s_attn = SpatialAtten(self.in_channels, self.channels) #空间注意力层
        self.fuse_attn = nn.Conv2d(self.channels*2, self.channels, 1, 1, 0) #融合注意力层
        self.heatmap_conv = nn.Conv2d(self.channels, self.out_channels, 1, 1, 0) #输出层

        self.heatmap_loss = FocalLoss()

        for m in self.modules(): #对模型中的卷积层和线性层的参数进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        self.prior_prob = 0.01 #偏置概率
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob) #用于初始化输出层的偏置
        self.heatmap_conv.bias.data.fill_(bias_value)

    def forward(self, features, instances): #特征张量、目标实例信息的字典
        global_features = self.conv_down(features) #1x1卷积层进行下采样
        instance_features = global_features[instances['instance_imgid']] #选择对应的目标实例的特征向量
        instance_params = instances['instance_param']
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params, instances['instance_coord']) #生成实例特征向量
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats), dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats) #融合特征向量
        cond_instance_feats = F.relu(cond_instance_feats) #进行激活

        pred_instance_heatmaps = _sigmoid(self.heatmap_conv(cond_instance_feats)) #通过一个1x1卷积层将融合后的特征映射到目标实例热力图

        if self.training:
            gt_instance_heatmaps = instances['instance_heatmap']
            gt_instance_masks = instances['instance_mask']
            single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps, gt_instance_heatmaps, gt_instance_masks)
            return single_heatmap_loss #返回单个热力图损失（用于反向传播）
        else:
            return pred_instance_heatmaps #预测的实例热力图

class ChannelAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size() #B表示batch size，C表示通道数，H和W分别表示特征图的高和宽
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1) #每个instance的特征参数
        return global_features * instance_params.expand_as(global_features)

class SpatialAtten(nn.Module):#空间层面，增大人体实例权重，减小背景权重
    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels) #一个线性层，用于将输入的实例参数转换为对应的特征权重
        self.feat_stride = 4 #表示图像特征图的步长
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2) #用于计算实例参数和位置信息的注意力掩码

    def forward(self, global_features, instance_params, instance_inds): #global_features表示输入的全局特征图，instance_params表示输入的实例参数，instance_inds表示输入的实例索引
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1) #将instance_params转换为对应的特征权重
        feats = global_features * instance_params.expand_as(global_features) #实例特征计算
        fsum = torch.sum(feats, dim=1, keepdim=True) #对实例特征进行求和得到输入特征（包括全局特征和相对坐标信息）
        input_feats = fsum
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)
        n_inst = instance_inds.size(0)
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1])
        instance_locations = instance_locations
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_feats = torch.cat((input_feats, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_features * mask

def compute_locations(h, w, stride, device): #计算特征图中每个像素对应于原图中的位置坐标
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    ) #生成等差数列x
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    ) #生成等差数列y
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) #生成这两个等差数列的笛卡尔积
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2 #得到每个特征图像素对应于原图中的中心点的坐标
    return locations