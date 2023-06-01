import os
import mmcv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import math


from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import RoIAlign
from mmdet.core import multi_apply
from mmdet.core import parsing_matrix_nms as matrix_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import AdaptBlock
from mmdet.core.utils import generate_coordinate
from mmdet.models.losses.cid_loss import FocalLoss, ContrastiveLoss

#from ..builder import DETECTORS
# from mmdet.models.utils import ConvModule, bias_init_with_prob
# AdaptBlock

INF = 1e8 #无穷大\
chn_per_part = 22

def get_heat_val(sigma, x, y, x0, y0, lambda_x = 1., lambda_y = 1.): #计算高斯分布函数值
    # TODO: make sure sigma float，sigma表示高斯函数的标准差
    if x!=x0 or y!=y0:
        #g = 0.32
        g = torch.exp((- (lambda_x * (x - x0) ** 2 + lambda_y * (y - y0) ** 2) / (2 * sigma ** 2)).float())
    else:
        g = 1
    return g

def get_heat_dist(sigma, h, w, x0, y0, w_m = 1,h_m = 1): #计算高斯分布函数在二维空间上的值

    x_range = torch.linspace(0, w-1, w, device=x0.device)
    y_range = torch.linspace(0, h-1, h, device=x0.device) #生成一个长度为w和h的一维Tensor，分别表示横向和纵向的坐标范围
    y, x = torch.meshgrid(y_range, x_range) #表示横向和纵向的坐标矩阵，得到一个大小为(h, w, 2)的Tensor，
    #其中第一个维度表示纵向坐标，第二个维度表示横向坐标，第三个维度是一个长度为2的Tensor，分别表示横向和纵向的坐标。
    
    if h_m == 0:
        g = torch.zeros([h, w], device=x0.device)
    else:
        lambda_x = 1
        lambda_y = w_m/h_m
        g = torch.exp((- (lambda_x * (x - x0) ** 2 + lambda_y * (y - y0) ** 2) / (2 * sigma ** 2)).float())

    return g

def center_of_mass(bitmasks, adapt_center=False): #计算二值掩模的质心（中心点）
    
    n, h, w = bitmasks.size() #bitmasks一个形状为 [n, h, w] 的张量，代表了 n 个大小为 h x w 的二值掩模
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device) #创建一个长度为h(w)的浮点数序列ys(xs)它们分别代表了掩模图像中y轴和x轴上的坐标。

    #计算每个掩模的重心的 x 和 y 坐标
    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1) #计算一阶和零阶矩
    center_x = m10 / m00 #计算中心坐标，形状为n的浮点数张量
    center_y = m01 / m00
    
    if adapt_center:
        # TODO
        # center_o_x = center_x
        # center_o_y = center_y
        dist_y = ((torch.ones_like(bitmasks)*ys[:, None]) - center_y.int().reshape(n,1,1))
        dist_x = ((torch.ones_like(bitmasks)*xs) - center_x.int().reshape(n,1,1))
        dist_maps = torch.sqrt(dist_y**2 + dist_x**2)*((~bitmasks.bool()).int()*(np.sqrt(h*h+w*w))+1)
        dists = dist_maps.reshape(n,-1)
        min_index = torch.argmin(dists,1)
        center_y = min_index // w
        center_x = min_index % w
        # if torch.tensor([True], device='cuda:0') in (center_o_x.int() != center_x):
        #     print("find adapt_center.")
        #     import pdb;pdb.set_trace()

    return center_x, center_y

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1) #得到每个像素周围 kernel x kernel 区域内的最大值
    keep = (hmax[:, :, :-1, :-1] == heat).float() #如果它是周围区域内的最大值之一，则将其保留，否则将其值设为0
    return heat * keep

def dice_loss(input, target): #张量input和target，分别表示模型的预测结果和真实结果
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1) #表示input和target的交集
    b = torch.sum(input * input, 1) + 0.001 #表示输入的平方和
    c = torch.sum(target * target, 1) + 0.001 #表示目标的平方和
    d = (2 * a) / (b + c) #表示 Dice 系数
    return 1-d #返回dice loss

class MaskFeatModule(BaseModule):
    """SOLOv2 mask feature map branch used in `SOLOv2: Dynamic and Fast
    Instance Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        in_channels (int): Number of channels in the input feature map.输入特征图的通道数
        feat_channels (int): Number of hidden channels of the mask feature 
             map branch.  mask feature map分支的隐藏通道数
        start_level (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.  用于预测mask feature map的起始特征图级别
        end_level (int): The ending feature map level from rpn that
             will be used to predict the mask feature map.  用于预测mask feature map的结束特征图级别。
        out_channels (int): Number of output channels of the mask feature  mask feature map分支的输出通道数。
             map branch. This is the channel count of the mask  
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): Downsample factor of the mask feature map output.
            Default: 4.   mask feature map输出的下采样因子,默认为4。
        conv_cfg (dict): Config dict for convolution layer. Default: None.  卷积层的配置字典,默认为None
        norm_cfg (dict): Config dict for normalization layer. Default: None.  卷积层的配置字典,默认为None
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,  #初始化模块，包括各个层的初始化和权重初始化
                 in_channels,
                 feat_channels,
                 start_level,
                 end_level,
                 out_channels,
                 mask_stride=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01)]):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.fp16_enabled = False
        self.init_weights()

    def _init_layers(self):  #初始化网络中的各层
        self.convs_all_levels = nn.ModuleList() #用于存储所有的卷积层和上采样层
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                convs_per_level.add_module(
                    f'conv{i}',
                    ConvModule( #创建卷积层conv0
                        self.in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.end_level:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    convs_per_level.add_module(
                        f'conv{j}',
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            inplace=False))
                    convs_per_level.add_module(
                        f'upsample{j}',
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
                    continue

                convs_per_level.add_module(
                    f'conv{j}',
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                convs_per_level.add_module(
                    f'upsample{j}',
                    nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            self.feat_channels,
            self.out_channels,
            1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self): #初始化网络的权重。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    @auto_fp16() #用于将输入特征和输出特征转换为半精度浮点数格式以提高模型的训练和推断速度。
    def forward(self, feats):
        inputs = feats[self.start_level:self.end_level + 1] #将输入的特征切片成start_level和end_level之间的层
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord_feat = generate_coordinate(input_p.size(),
                                                 input_p.device) #生成一个坐标特征，并与输入特征进行拼接
                input_p = torch.cat([input_p, coord_feat], 1)

            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            feature_add_all_level = feature_add_all_level + \
                self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level) #在conv_pred层中进行卷积
        return feature_pred

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
        self.output_res = cfg.MODEL.IIA.OUT_RES
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.IIA.IN_CHANNELS
        self.out_channels = cfg.MODEL.IIA.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints + 1 #判断模型输出的通道数是否等于关键点的数量加上一个背景通道
        self.prior_prob = cfg.MODEL.BIAS_PROB

        self.keypoint_center_conv = nn.Conv2d(self.in_channels, 1, 1, 1, 0) #计算关键点中心热力图，1x1 的卷积核，步长为 1，填充为 0
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

    def heatmap_generation(self, batch_inputs, gaussian=False):
        hms = np.zeros((1, self.output_res, self.output_res), dtype=np.float32)
        bboxes = batch_inputs['instance_bboxes']
        import pdb;pdb.set_trace()
        if gaussian:
            for bbox in bboxes:
                h = np.sqrt(np.power(bbox[2, 0]-bbox[0, 0], 2) + np.power(bbox[2, 1] - bbox[0, 1], 2))
                w = np.sqrt(np.power(bbox[1, 0]-bbox[0, 0], 2) + np.power(bbox[1, 1] - bbox[0, 1], 2))
                radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
        return hms
    
    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian2D(self, shape, sigma=1.):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def forward(self, features, batch_inputs=None):
        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features)) #iia预测值：sigmoid函数放缩到0~1之间

        if self.training: #如果模型处于训练模式，则使用batch_inputs对象执行训练；否则，返回一个instances对象，该对象包含模型对输入的预测
            # multi_heatmap: [1, H, W] heatmap, 每个关键点一张HW图
            center_labels = [
                torch.cat([center_labels_level_img.flatten()
                        for center_labels_level_img in center_labels_level])
                for center_labels_level in zip(*batch_inputs[''])
            ]
            flatten_center_labels = torch.cat(center_labels)
            
            num_ins = (flatten_center_labels==0).sum()

            flatten_center_preds = pred_multi_heatmap.permute(0, 2, 3, 1).reshape(-1, 1)
            
            loss_center = batch_inputs['center_loss_function'](flatten_center_preds, flatten_center_labels, avg_factor=num_ins + 1)


            self.heatmap_generation(batch_inputs)

            gt_multi_heatmap = [x['multi_heatmap'].unsqueeze(0).to(self.device) for x in batch_inputs]
            gt_multi_heatmap = torch.cat(gt_multi_heatmap, dim=0)
            # gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(self.device) for x in batch_inputs]
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
                instance_param = self._sample_batch_inputs(features[i], instance_coord)
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
                instance_param = self._sample_batch_inputs(features[0], instance_coord)
                instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device)
                if self.flip_test:
                    instance_param_flip = self._sample_batch_inputs(features[1], instance_coord)
                    instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                    instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
                    instance_param = torch.cat((instance_param, instance_param_flip), dim=0)
                    instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)

                instances['instance_coord'] = instance_coord
                instances['instance_imgid'] = instance_imgid
                instances['instance_param'] = instance_param
                instances['instance_score'] = scores

            return instances
    
    def _sample_batch_inputs(self, features, pos_ind): #pos_ind是一个二维数组，表示特征图中需要提取特征的位置
        batch_inputs = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return batch_inputs.permute(1, 0) #将维度从 (channels, batch_size) 转换为 (batch_size, channels)

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
        c_instance_batch_inputs = self.c_attn(instance_features, instance_params)
        s_instance_batch_inputs = self.s_attn(instance_features, instance_params, instances['instance_coord']) #生成实例特征向量
        cond_instance_batch_inputs = torch.cat((c_instance_batch_inputs, s_instance_batch_inputs), dim=1)
        cond_instance_batch_inputs = self.fuse_attn(cond_instance_batch_inputs) #融合特征向量
        cond_instance_batch_inputs = F.relu(cond_instance_batch_inputs) #进行激活

        pred_instance_heatmaps = _sigmoid(self.heatmap_conv(cond_instance_batch_inputs)) #通过一个1x1卷积层将融合后的特征映射到目标实例热力图

        if self.training:
            gt_instance_heatmaps = instances['instance_heatmap']
            gt_instance_masks = instances['instance_mask']
            single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps, gt_instance_heatmaps, gt_instance_masks)
            return single_heatmap_loss #返回单个热力图损失（用于反向传播）
        else:
            return pred_instance_heatmaps #预测的实例热力图

class ChannelAtten(nn.Module):  #对输入的全局特征图和每个实例的特征参数进行通道级的注意力机制计算
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
        batch_inputs = global_features * instance_params.expand_as(global_features) #实例特征计算
        fsum = torch.sum(batch_inputs, dim=1, keepdim=True) #对实例特征进行求和得到输入特征（包括全局特征和相对坐标信息）
        input_batch_inputs = fsum
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)
        n_inst = instance_inds.size(0)
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1])
        instance_locations = instance_locations
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_batch_inputs = torch.cat((input_batch_inputs, relative_coords), dim=1)
        mask = self.conv(input_batch_inputs).sigmoid()
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



@HEADS.register_module()
class CID_Parsing_basic(nn.Module):

    def __init__(self,
                 cfg,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 enable_center=True,  #IIA
                 enable_offset=True,  #GFD
                 enable_segment=False,
                 enable_heatmaploss=False,
                 enable_kernel_cate=False,
                 enable_ori_grid=False,
                 enhance_cate=None,
                 enable_moi=False,
                 enable_ignore=False,
                 enable_cate_eval=False,
                 enable_cate_decouple=False,
                 enable_keypoints=False,
                 enable_adapt_center=False,
                 mask_feature_head=None,
                 loss_ins=None,
                 loss_cate=None,
                 loss_center=None,
                 loss_offset=None,
                 loss_refine=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None,
                 **kwargs):
        super(CID_Parsing_basic, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_cate = build_loss(loss_cate)
        self.loss_center = build_loss(loss_center)
        self.loss_offset = build_loss(loss_offset)
        self.loss_refine = build_loss(loss_refine) if loss_refine else None
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.enable_offset = enable_offset
        self.enable_center = enable_center
        self.enable_segment = enable_segment
        self.enable_heatmaploss = enable_heatmaploss
        self.enable_kernel_concat = enable_kernel_cate
        self.enable_ori_grid = enable_ori_grid
        self.enhance_cate = enhance_cate
        self.enable_ignore = enable_ignore
        self.enable_moi = enable_moi
        self.enable_cate_decouple = enable_cate_decouple
        self.enable_cate_eval = enable_cate_eval
        self.enable_keypoints = enable_keypoints
        self.enable_adapt_center = enable_adapt_center
        self.mask_feature_head = MaskFeatModule(**mask_feature_head)
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self.chn_per_part = chn_per_part
        self._init_layers()
        self.iia = build_iia_module(cfg)
        self.gfd = build_gfd_module(cfg)
        #self.backbone = builder.build_backbone(backbone)

        self.multi_heatmap_loss_weight = cfg.LOSS.MULTI_HEATMAP_LOSS_WEIGHT
        self.contrastive_loss_weight = cfg.LOSS.CONTRASTIVE_LOSS_WEIGHT
        self.single_heatmap_loss_weight = cfg.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT

        # inference
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = cfg.DATASET.FLIP_INDEX
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
    #模型初始化，创建模型

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList() #类别卷积层
        self.kernel_convs = nn.ModuleList() #核卷积层
        if self.enable_center == True:
            self.center_convs = nn.ModuleList() #中心卷积层
        if self.enable_offset == True:
            self.offset_blocks = nn.ModuleList() #偏移卷积层
            self.offset_final = nn.ModuleList() #偏移卷积层的输出卷积层
        if self.enable_segment == True:
            self.segment_convs = nn.ModuleList() #分割卷积层
        if self.enable_cate_decouple == True:
            self.classify_convs = nn.ModuleList() #分类卷积层
        if self.enable_keypoints == True:
            self.keypoints_convs = nn.ModuleList() #关键点卷积层
        
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
                
            if self.enable_segment == True:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.segment_convs.append(
                    ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            cate_kernel = 3
            cate_dilation = 1
            cate_padding = 1
            if self.enhance_cate:
                if 'large_kernel' in self.enhance_cate:
                    cate_kernel = 5
                if 'dilation' in self.enhance_cate:
                    cate_dilation = 2
                cate_padding = int(cate_dilation*((cate_kernel-1)/2))
                if 'deep_branch' in self.enhance_cate:
                    if i == self.stacked_convs - 1:
                        for i in range(2):
                            self.cate_convs.append(
                                ConvModule(
                                    chn,
                                    self.seg_feat_channels,
                                    cate_kernel,
                                    stride=1,
                                    padding=cate_padding,
                                    dilation=cate_dilation,
                                    conv_cfg=cfg_conv,
                                    norm_cfg=norm_cfg,
                                    bias=norm_cfg is None)) 
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    cate_kernel,
                    stride=1,
                    padding=cate_padding,
                    dilation=cate_dilation,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            if self.enable_center == True:
                chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                self.center_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
            if self.enable_cate_decouple == True:
                chn = self.in_channels if i == 0 else self.seg_feat_channels
                self.classify_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
            if self.enable_keypoints == True:
                chn = self.in_channels if i == 0 else self.seg_feat_channels
                self.keypoints_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))

        if self.enable_moi == True:
            self.roi_extractor = RoIAlign((32,32),
                                            spatial_scale = 1/4,
                                            sample_num=2,
                                            use_torchvision=True)
            self.class_conv1 = ConvModule(
                    258,
                    512,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None)
            self.class_conv2 = ConvModule(
                    512,
                    256,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None)
            self.class_fc1 = nn.Linear(16384, 1024)
            self.class_fc2 = nn.Linear(1024, 58)
            self.relu = nn.ReLU(inplace=True)

        if self.enable_center == True:
            self.dekr_convs = nn.Conv2d(
                self.in_channels * 5, self.in_channels, 3, padding=1)
            self.center_heatmap = nn.Conv2d(
                self.seg_feat_channels, 1, 3, padding=1)

        if self.enable_offset == True:
            self.offset_convs = nn.Conv2d(
                self.in_channels * 5 + 2, self.cate_out_channels*22, 3, padding=1)
            for i in range(self.cate_out_channels):
                self.offset_blocks.append(
                    AdaptBlock(
                        22,
                        22,
                        1
                    )
                )
                self.offset_final.append(
                        nn.Conv2d(
                            in_channels=22,
                            out_channels=2,
                            kernel_size=1,
                            stride=1,
                            padding=0
                    )
                )
        if self.enable_segment == True:
            self.segment_converts = nn.Conv2d(
                self.in_channels * 5, self.in_channels, 3, padding=1)
            self.segment_kernel = nn.Conv2d(
                self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)
        
        if self.enable_cate_decouple == True:
            self.solo_class = nn.Conv2d(
                self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
        if self.enable_keypoints == True:
            self.keypoints_heat = nn.Conv2d(
            self.seg_feat_channels, 20, 3, padding=1)
            self.key_trans_convs = nn.Conv2d(
                self.in_channels * 5, self.in_channels, 3, padding=1)

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    def init_weights(self): #对模型中各个部分的权重进行初始化，采用正态分布或者均匀分布等随机方法来生成初始的参数值
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)
        
        if self.enable_segment == True: #是否启用分割
            for m in self.segment_convs:
                normal_init(m.conv, std=0.01)
            normal_init(self.segment_kernel, std=0.01)
            normal_init(self.segment_converts, std=0.01)

        if self.enable_cate_decouple == True: #是否启用类别解耦
            for m in self.classify_convs: #初始化分类卷积
                normal_init(m.conv, std=0.01)
            normal_init(self.solo_class, std=0.01, bias=bias_cate) #初始化分类器的类别部分
            normal_init(self.key_trans_convs, std=0.01, bias=bias_Cate) #初始化关键点转换卷积

        if self.enable_center == True: #启用中心点
            for m in self.center_convs:
                normal_init(m.conv, std=0.01)
            normal_init(self.center_heatmap, std=0.01, bias=bias_cate) #初始化中心点热图
            normal_init(self.dekr_convs, std=0.01) #初始化关键点解码器卷积
        
        if self.enable_keypoints == True:
            for m in self.keypoints_convs: #关键点卷积
                normal_init(m.conv, std=0.01)
            normal_init(self.keypoints_heat, std=0.01,bias=bias_cate) #关键点热图

        if self.enable_offset == True:
            normal_init(self.offset_convs, std=0.01) #偏移卷积
            for m in self.offset_final: #偏移最终坐标
                normal_init(m, std=0.01)
            for m in self.offset_blocks: #偏移块
                if hasattr(m, 'transform_matrix_conv'):
                    nn.init.constant_(m.transform_matrix_conv.weight, 0)
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.transform_matrix_conv.bias, 0)
                if hasattr(m, 'translation_conv'):
                    nn.init.constant_(m.translation_conv.weight, 0)
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.translation_conv.bias, 0)
        
        if self.enable_moi == True: #多物体实例分割，分类卷积1、分类卷积2、分类器全连接层
            normal_init(self.class_conv1.conv, std=0.01)
            normal_init(self.class_conv2.conv, std=0.01)
            nn.init.xavier_uniform_(self.class_fc1.weight)
            nn.init.constant_(self.class_fc1.bias, 0)
            nn.init.xavier_uniform_(self.class_fc2.weight)
            nn.init.constant_(self.class_fc2.bias, 0)
            
    def forward(self, feats, gt_inputs):
        
        #img = [x['image'].unsqueeze(0).to(self.device) for x in batch_inputs] #按顺序取出图片
        #img = torch.cat(img, dim=0) #concat将图片进行拼接
        # feats = self.backbone(images) #将图片输入backbone

        # mask feature branch
        mask_feats = self.mask_feature_head(feats)
        mask_feat_size = mask_feats.shape[-2:]
              
        # center branch & offset branch & segment branch
        new_feats = self.concate_feats(feats) #将所有特征层拼接在一起，送入中心分支、偏移分支、分割分支和关键点分支进行处理(送入IIA和GFD)
        mixed_feats = torch.cat(new_feats, dim=1) # size为[2,1280,184,264]
        # mixed_feats.shape: (N, 5*C_in, H, W)
        
        inst_coord = []
        inst_coords = []
        
        gt_masks_pt = gt_inputs[2]
        for i in range(len(gt_masks_pt)):
            gt_masks = gt_masks_pt[i]
            
            center_x, center_y = center_of_mass(torch.from_numpy(gt_masks), adapt_center=False) #实例坐标,缺一个参数！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            inst_coords.append(np.array([np.array(center_y), np.array(center_x)]))
        
        
        if len(inst_coords) > 0:
            inst_coords = np.concatenate(inst_coords, axis=0) #将实例的坐标数据拼接成一个数组
        
        batch_inputs = {}
        batch_inputs['instance_coord'] = torch.from_numpy(inst_coords)
        batch_inputs['instance_bboxes'] = gt_inputs[0]
        
        center_label_list, ins_label_list, ins_ind_label_list, grid_order_list  = multi_apply(
            self.parsing_center_target_single,
            gt_inputs[0], 
            gt_inputs[1], 
            gt_inputs[2], 
            gt_inputs[3],
            [self.enable_heatmaploss]*len(gt_inputs[1]),
            gt_inputs[6],
            mask_feat_size = mask_feat_size)

        batch_inputs['center_heatmap'] = center_label_list
        batch_inputs['center_loss_function'] = self.loss_center
        

        if self.training:
            multi_heatmap_loss, contrastive_loss, instances = self.iia(mixed_feats, batch_inputs) #输入到IIA

            # limit max instances in training
            if 0 <= self.max_instances < instances['instance_param'].size(0):
                inds = torch.randperm(instances['instance_param'].size(0), device=self.device).long()
                for k, v in instances.items():
                    instances[k] = v[inds[:self.max_instances]]

            single_heatmap_loss = self.gfd(feats, instances) #输入到gfd

            losses = {}
            losses.update({'multi_heatmap_loss': multi_heatmap_loss * self.multi_heatmap_loss_weight})
            losses.update({'single_heatmap_loss': single_heatmap_loss * self.single_heatmap_loss_weight})
            losses.update({'contrastive_loss': contrastive_loss * self.contrastive_loss_weight}) #得到3个loss并返回
            return losses
        else:
            results = {}
            if self.flip_test:
                feats[1, :, :, :] = feats[1, :, :, :].flip([2])

            instances = self.iia(feats)
            if len(instances) == 0: return results

            instance_heatmaps = self.gfd(feats, instances)

            if self.flip_test:
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = instance_heatmaps_flip[:, self.flip_index, :, :]
                instance_heatmaps = (instance_heatmaps + instance_heatmaps_flip) / 2.0

            instance_scores = instances['instance_score']
            num_people, num_keypoints, h, w = instance_heatmaps.size()
            center_pool = F.avg_pool2d(instance_heatmaps, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)
            instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
            nms_instance_heatmaps = instance_heatmaps.view(num_people, num_keypoints, -1)
            vals, inds = torch.max(nms_instance_heatmaps, dim=2)
            x, y = inds % w, (inds / w).long()
            # shift coords by 0.25
            x, y = self.adjust(x, y, instance_heatmaps)
            
            vals = vals * instance_scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)

            poses[:, :, :2] = poses[:, :, :2] * 4 + 2
            scores = torch.mean(poses[:, :, 2], dim=1)

            results.update({'poses': poses})
            results.update({'scores': scores})

            return results,  mask_feats
    
    def adjust(self, res_x, res_y, heatmaps): #预测坐标和热图作为输入
        n, k, h, w = heatmaps.size()#[2:] #n 是批量大小，k 是关键点的数量，h 和 w 分别是热图的高度和宽度。

        #计算预测坐标周围的四个相邻坐标
        x_l, x_r = (res_x - 1).clamp(min=0), (res_x + 1).clamp(max=w-1)
        y_t, y_b = (res_y + 1).clamp(max=h-1), (res_y - 1).clamp(min=0)
        n_inds = torch.arange(n)[:, None].to(self.device)
        k_inds = torch.arange(k)[None].to(self.device)

        #计算 x 和 y 方向（px 和 py）的偏移量
        px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] - heatmaps[n_inds, k_inds, res_y, x_l])*0.25
        py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] - heatmaps[n_inds, k_inds, y_b, res_x])*0.25

        res_x, res_y = res_x.float(), res_y.float()
        x_l, x_r = x_l.float(), x_r.float()
        y_b, y_t = y_b.float(), y_t.float()
        px = px*torch.sign(res_x-x_l)*torch.sign(x_r-res_x)
        py = py*torch.sign(res_y-y_b)*torch.sign(y_t-res_y)

        res_x = res_x.float() + px
        res_y = res_y.float() + py

        return res_x, res_y #函数返回调整后的坐标

    def split_feats(self, feats): #对特征图进行预处理，feats是一个包含了 5 个特征图的元组
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear',align_corners=True), #被 resize 成原来的一半大小
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear',align_corners=True))

    def concate_feats(self, feats):
        return (feats[0], #mode='bilinear'表示使用双线性插值法，align_corners=True表示是否对齐角点
                F.interpolate(feats[1], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[2], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[3], size=feats[0].shape[-2:], mode='bilinear',align_corners=True),
                F.interpolate(feats[4], size=feats[0].shape[-2:], mode='bilinear',align_corners=True))

    def forward_center(self, x, eval=False):
         # reduce dimension by 1*1 conv
        center_feat = self.dekr_convs(x) #降低张量x的维度
        inst_coords=[]
        inst_coord = []

        #concat coord
        x_range = torch.linspace(-1, 1, center_feat.shape[-1], device=center_feat.device)
        y_range = torch.linspace(-1, 1, center_feat.shape[-2], device=center_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([center_feat.shape[0], 1, -1, -1])
        x = x.expand([center_feat.shape[0], 1, -1, -1])
       
        inst_coord.append([y, x])
        inst_coords.append(np.array(inst_coord))
        return inst_coords
    
    def forward_keypoints(self, x, eval=False): #未用
        # reduce dimension by 1*1 conv降维
        keypoints_feat = self.key_trans_convs(x) #关键点特征

        # resize to grids
        if self.enable_ori_grid:
            seg_num_grid = self.seg_num_grids[0]
        else:
            seg_num_grid = self.seg_num_grids[-2]
        keypoints_feat = F.interpolate(keypoints_feat, size=seg_num_grid, mode='bilinear',align_corners=True) #双线性插值法调整keypoints_feat的大小

        # get heatmap
        keypoints_feat = keypoints_feat.contiguous()
        for i, keypoints_layer in enumerate(self.keypoints_convs):
            keypoints_feat = keypoints_layer(keypoints_feat)
        keypoints_pred = self.keypoints_heat(keypoints_feat) #进行卷积操作得到关键点预测的热力图

        if eval:
            keypoints_pred = points_nms(keypoints_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1) #使用非极大值抑制方法对预测热力图进行后处理
        return keypoints_pred

    def forward_segment(self, x): #预测分割的掩模
        ins_kernel_feat = self.segment_converts(x)
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device) #将x和y的坐标范围映射到[-1,1]
        y, x = torch.meshgrid(y_range, x_range) #生成与坐标点对应的网格点
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1) #生成的坐标点与网格点按维度拼接
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1) #与ins_kernel_feat张量按通道维度拼接

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[0]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear',align_corners=True) #与ins_kernel_feat张量按通道维度拼接

        cate_feat = kernel_feat[:, :-2, :, :] #将cate_feat设置为kernel_feat除最后两个通道以外的所有通道

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.segment_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.segment_kernel(kernel_feat) #对kernel_feat张量进行卷积操作，得到kernel_pred张量，它是分割的掩模输出

        return kernel_pred

    def forward_classify(self, x, idx, eval=False, upsampled_size=None): #用于分类，idx参数是用于指示使用哪种分割策略的指标，eval为True，则执行测试模式，否则执行训练模式
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device) ##将x和y的坐标范围映射到[-1,1]
        y, x = torch.meshgrid(y_range, x_range) ##生成与坐标点对应的网格点
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1) #生成的坐标点与网格点按维度拼接
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1) #与ins_kernel_feat张量按通道维度拼接
        
        # class branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear',align_corners=True) #插值处理

        cate_feat = kernel_feat[:, :-2, :, :]

        # class branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.classify_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_class(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1) # 进行非极大值抑制
        return cate_pred

    def get_guassian_cate(self, cate_label_list): #用于生成类别预测的高斯热力图
        cate_label_list_p = []
        sigma = 2
        for imgs_idx in range(len(cate_label_list)):
            cate_label_list_p_img = []
            for level_img_idx in range(len(cate_label_list[imgs_idx])):
                cate_label_list_p_img_level = []
                w, h = cate_label_list[imgs_idx][level_img_idx].shape
                level_img = cate_label_list[imgs_idx][level_img_idx]
                for p_i in range(self.cate_out_channels):
                    img_p_fg = (level_img == (p_i+1)).float()
                    img_p = img_p_fg
                    for pt in img_p_fg.nonzero():
                        x0 = pt[0]
                        y0 = pt[1]
                        ul = int(np.floor(x0.cpu().numpy() - 3 * sigma - 1)
                            ), int(np.floor(y0.cpu().numpy() - 3 * sigma - 1))
                        br = int(np.ceil(x0.cpu().numpy() + 3 * sigma + 2)
                                ), int(np.ceil(y0.cpu().numpy() + 3 * sigma + 2))

                        cc, dd = max(0, ul[0]), min(br[0], w)
                        aa, bb = max(0, ul[1]), min(br[1], h)

                        #import pdb;pdb.set_trace()
                        joint_rg = get_heat_dist(sigma,bb-aa,dd-cc,x0-cc,y0-aa)

                        # joint_rg = np.zeros((bb-aa, dd-cc))
                        # for sy in range(aa, bb):
                        #     for sx in range(cc, dd):
                        #         joint_rg[sy-aa, sx -
                        #                 cc] = get_heat_val(sigma, sx, sy, x0, y0)

                        img_p[aa:bb, cc:dd] = torch.from_numpy(np.maximum(
                            img_p[aa:bb, cc:dd].cpu().numpy(), joint_rg.cpu().numpy()))

                    cate_label_list_p_img_level.append(img_p.unsqueeze(0))
                cate_label_list_p_img_level = torch.cat(cate_label_list_p_img_level)
                cate_label_list_p_img.append(cate_label_list_p_img_level.unsqueeze(0))
            cate_label_list_p_img = torch.cat(cate_label_list_p_img)
            cate_label_list_p.append(cate_label_list_p_img.unsqueeze(0))
        cate_label_list_p = torch.cat(cate_label_list_p)
        return cate_label_list_p

    def loss(self,
             cate_preds,
             kernel_preds,
             center_preds, 
             offset_preds,
             segment_preds,
             classify_preds,
             keypoints_preds,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             gt_parsing_list,
             gt_semantic_seg_list,
             gt_keypoints_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        mask_feat_size = ins_pred.size()[-2:]
        self.img_metas = img_metas
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, classify_label_list, parsing_label_list = multi_apply(
            self.parsing_category_target_single,
            gt_bbox_list, 
            gt_label_list, 
            gt_mask_list, 
            gt_parsing_list,
            [self.enable_heatmaploss]*len(gt_label_list),
            mask_feat_size=mask_feat_size,
            adapt_center=self.enable_adapt_center)
        
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred_list = []
        ins_refine_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            b_ins_refine_list = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)                  
                b_mask_pred.append(cur_ins_pred)
                b_ins_refine_list = b_ins_refine_list + [idx]*I
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)
            ins_refine_list = ins_refine_list + b_ins_refine_list

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []

        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        if self.enable_ignore == True:
            loss_ins = torch.tensor(0,dtype=torch.float32, device=kernel_pred.device)

        # cate
        ## Apply Focal Loss
        if self.enable_heatmaploss == False:
            try:
                cate_labels = [
                    torch.cat([cate_labels_level_img.flatten()
                            for cate_labels_level_img in cate_labels_level])
                    for cate_labels_level in zip(*cate_label_list)
                ]
                flatten_cate_labels = torch.cat(cate_labels)
                cate_preds = [
                    cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
                    for cate_pred in cate_preds
                ]
                assert cate_labels.__len__()  == cate_preds.__len__()
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            flatten_cate_preds = torch.cat(cate_preds)
            #import pdb;pdb.set_trace()
            # flatten_cate_preds[flatten_cate_labels>0] = torch.exp(F.log_softmax(flatten_cate_preds[flatten_cate_labels>0],dim=1))
            # flatten_cate_preds = (flatten_cate_preds - 0.5)*10
            loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        
        ## Apply Heatmap Loss
        ### add heatmap loss like DEKR
        if self.enable_heatmaploss == True:

            ### generate parsing_mask
            cate_label_list_p = self.get_guassian_cate(cate_label_list)
        
            cate_labels = cate_label_list_p.permute(1,0,3,4,2)
            cate_labels = [
                cate_label.reshape(-1, self.cate_out_channels)
                for cate_label in cate_labels
            ]

            cate_preds = [
                cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
                for cate_pred in cate_preds
            ]
            #import pdb;pdb.set_trace()
            flatten_cate_labels = torch.cat(cate_labels)
            flatten_cate_preds = torch.cat(cate_preds)

            loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, (flatten_cate_labels==0).float())

        if self.enable_moi == True:
            
            ins_labels = torch.cat([ins_label for ins_label in ins_labels])
            ins_preds = []
            for ins_preds_l in ins_pred_list:
                if ins_preds_l is not None:
                    ins_preds.append(ins_preds_l)
            ins_preds = torch.cat([ins_pred_i for ins_pred_i in ins_preds])
            
            mois = []
            moi_feats = []
            moi_feats_tmp = []
            memory_limit = 50
            for ins_idx in range(len(ins_labels)):
                try:
                    ys, xs = torch.where(ins_labels[ins_idx])
                    l,t = xs.min(), ys.min()
                    r,b = xs.max(), ys.max()
                    center_x = (r+l)//2
                    center_y = (t+b)//2
                    mois.append(torch.tensor([l,t,r,b], dtype=torch.float32,device=t.device).unsqueeze(0))
                    parsing_feat = ins_labels[ins_idx]
                    base_feat = ins_pred[ins_refine_list[ins_idx]]
                    conv_feat = ins_preds[ins_refine_list[ins_idx]]
                    moi_feat = torch.cat([parsing_feat.float().unsqueeze(0), base_feat, conv_feat.unsqueeze(0)])
                    moi_feats_tmp.append(moi_feat.unsqueeze(0))
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
                if memory_limit != 0 and ins_idx != (len(ins_labels)-1):
                    memory_limit -= 1
                else:
                    moi_feats.append(self.roi_extractor(torch.cat(moi_feats_tmp),mois))
                    moi_feats_tmp = []
                    mois = []
                    memory_limit = 50
            moi_feats = torch.cat(moi_feats).detach()
            moi_feats = self.class_conv1(moi_feats)
            moi_feats = self.class_conv2(moi_feats).view(moi_feats.shape[0],-1)
            
            moi_feats = self.relu(self.class_fc1(moi_feats))
            moi_feats = self.relu(self.class_fc2(moi_feats))
            #import pdb;pdb.set_trace()
            parsing_labels = torch.cat([torch.cat([torch.tensor(ins_labels_level_img).cuda() 
                                            for ins_labels_level_img in ins_labels_level], 0).long().cuda() 
                                                for ins_labels_level in zip(*parsing_label_list)])
            loss_refine = self.loss_refine(moi_feats, parsing_labels, avg_factor=moi_feats.shape[0] + 1)
        else:
            loss_refine = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)     

        if self.enable_center == True:
            # center
            ## Apply Focal Loss

            center_label_list, ins_label_list, ins_ind_label_list, grid_order_list  = multi_apply(
                self.parsing_center_target_single,
                gt_bbox_list, 
                gt_label_list, 
                gt_mask_list, 
                gt_parsing_list,
                [self.enable_heatmaploss]*len(gt_label_list),
                img_metas,
                mask_feat_size = mask_feat_size)
            
            if self.enable_heatmaploss == False:

                center_labels = [
                    torch.cat([center_labels_level_img.flatten()
                            for center_labels_level_img in center_labels_level])
                    for center_labels_level in zip(*center_label_list)
                ]
                flatten_center_labels = torch.cat(center_labels)
                
                num_ins = (flatten_center_labels==0).sum()

                flatten_center_preds = center_preds.permute(0, 2, 3, 1).reshape(-1, 1)
                
                loss_center = self.loss_center(flatten_center_preds, flatten_center_labels, avg_factor=num_ins + 1)

            # center
            ## Apply Heatmap Loss
            else:
                sigma=2
                for imgs_idx in range(len(center_label_list)):
                    for level_img_idx in range(len(center_label_list[imgs_idx])):
                        w, h = center_label_list[imgs_idx][level_img_idx].shape
                        level_img = center_label_list[imgs_idx][level_img_idx]
                        center_label_list[imgs_idx][level_img_idx] = center_label_list[imgs_idx][level_img_idx].float()
                        img_c_fg = (level_img==1)
                        for pt in img_c_fg.nonzero():
                            x0 = pt[0]
                            y0 = pt[1]
                            ul = int(np.floor(x0.cpu().numpy() - 6 * sigma - 1)
                             ), int(np.floor(y0.cpu().numpy() - 6 * sigma - 1))
                            br = int(np.ceil(x0.cpu().numpy() + 6 * sigma + 2)
                                    ), int(np.ceil(y0.cpu().numpy() + 6 * sigma + 2))

                            cc, dd = max(0, ul[0]), min(br[0], w)
                            aa, bb = max(0, ul[1]), min(br[1], h)

                            joint_rg = get_heat_dist(2*sigma,bb-aa,dd-cc,x0-cc,y0-aa)

                            # joint_rg = np.zeros((bb-aa, dd-cc))
                            # for sy in range(aa, bb):
                            #     for sx in range(cc, dd):
                            #         joint_rg[sy-aa, sx -
                            #                 cc] = get_heat_val(sigma, sx, sy, x0, y0)
                            center_label_list[imgs_idx][level_img_idx][aa:bb, cc:dd] = torch.from_numpy(np.maximum(
                                center_label_list[imgs_idx][level_img_idx][aa:bb, cc:dd].cpu().numpy(), joint_rg.cpu().numpy()))

                center_labels = [
                    torch.cat([center_labels_level_img.flatten()
                            for center_labels_level_img in center_labels_level])
                    for center_labels_level in zip(*center_label_list)
                ]

                flatten_center_labels = torch.cat(center_labels).unsqueeze(1)

                flatten_center_preds = center_preds.permute(0, 2, 3, 1).reshape(-1, 1)

                loss_center = self.loss_center(flatten_center_preds, flatten_center_labels, (flatten_center_labels==0).float())
        else:
            loss_center = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)
   

        # segment
        if self.enable_segment == True:
            # ins
            ins_labels = [torch.cat([ins_labels_level_img
                                    for ins_labels_level_img in ins_labels_level], 0)
                        for ins_labels_level in zip(*ins_label_list)]
            
            segment_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                            for kernel_preds_level_img, grid_orders_level_img in
                            zip(kernel_preds_level, grid_orders_level)]
                            for kernel_preds_level, grid_orders_level in zip([segment_preds], zip(*grid_order_list))]
            # generate masks
            seg_pred_list = []
            for b_kernel_pred in segment_preds:
                b_mask_pred = []
                for idx, seg_kernel_pred in enumerate(b_kernel_pred):
                    #import pdb;pdb.set_trace()
                    if seg_kernel_pred.size()[-1] == 0:
                        continue
                    cur_ins_pred = ins_pred[idx, ...]
                    H, W = cur_ins_pred.shape[-2:]
                    N, I = seg_kernel_pred.shape
                    cur_ins_pred = cur_ins_pred.unsqueeze(0)
                    seg_kernel_pred = seg_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    cur_ins_pred = F.conv2d(cur_ins_pred, seg_kernel_pred, stride=1).view(-1, H, W)
                    b_mask_pred.append(cur_ins_pred)
                if len(b_mask_pred) == 0:
                    b_mask_pred = None
                else:
                    b_mask_pred = torch.cat(b_mask_pred, 0)
                seg_pred_list.append(b_mask_pred)
            #import pdb;pdb.set_trace()
            ins_ind_labels = [
                torch.cat([ins_ind_labels_level_img.flatten()
                        for ins_ind_labels_level_img in ins_ind_labels_level])
                for ins_ind_labels_level in zip(*ins_ind_label_list)
            ]
            flatten_ins_ind_labels = torch.cat(ins_ind_labels)

            num_ins = flatten_ins_ind_labels.sum()

            # dice loss
            loss_seg = []
            for input, target in zip(seg_pred_list, ins_labels):
                if input is None:
                    continue
                input = torch.sigmoid(input)
                #import pdb;pdb.set_trace()
                loss_seg.append(dice_loss(input, target))
            loss_seg = torch.cat(loss_seg).mean()
            loss_seg = loss_seg * self.ins_loss_weight
        else:
            loss_seg = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)
        # offset
        if self.enable_offset == True: 
            # get offset map label
            # offset_map_label.shape:  (58, imgs, [2, S, S])
            # offset_pred.shape: [58 *[imgs, 2, S, S]]
            # offset_map_label.shape:  (imgs, 58, [2, S, S])
            # offset_pred.shape: [58 *[imgs, 2, S, S]]

            offset_map_label, weight_map_label  = multi_apply(
                self.parsing_offset_target_single,
                gt_bbox_list, 
                gt_label_list, 
                gt_mask_list, 
                gt_parsing_list,
                img_metas,
                mask_feat_size = mask_feat_size)
            
            offset_preds = torch.cat([torch.unsqueeze(offset_pred, dim=0) for offset_pred in offset_preds])

            offset_map_label = list(offset_map_label)
            weight_map_label = list(weight_map_label)
            #for p_i in range(self.cate_out_channels):
            #    offset_map_label[p_i] = torch.tensor(offset_map_label[p_i], device=offset_preds.device)
            #    weight_map_label[p_i] = torch.tensor(weight_map_label[p_i], device=offset_preds.device)
            for p_i in range(len(weight_map_label)):
                offset_map_label[p_i] = torch.tensor(offset_map_label[p_i], device=offset_preds.device)
                weight_map_label[p_i] = torch.tensor(weight_map_label[p_i], device=offset_preds.device)
            
            offset_map_label = torch.cat([torch.unsqueeze(offset_map, dim=0) for offset_map in offset_map_label]).permute(1,0,2,3,4)
            weight_map_label = torch.cat([torch.unsqueeze(weight_map, dim=0) for weight_map in weight_map_label]).permute(1,0,2,3,4)
            
            #self.show_offset_gt(offset_map_label, offset_preds, img_metas)

            offset_map_label = offset_map_label.reshape(-1, 1)
            weight_map_label = weight_map_label.reshape(-1, 1)
            
            offset_preds = offset_preds.reshape(-1, 1)

            num_ins = (weight_map_label>0).sum()
            loss_offset = self.loss_offset(offset_preds, offset_map_label, weight_map_label, avg_factor= num_ins)
        else:
            loss_offset = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)

        if self.enable_cate_decouple == True:
            try:
                classify_labels = [
                    torch.cat([cate_labels_level_img.flatten()
                            for cate_labels_level_img in cate_labels_img])
                    for cate_labels_img in zip(*classify_label_list)
                ]
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            
            flatten_classify_labels = torch.cat(classify_labels)
            
            classify_preds_temp = []
            for classify_pred_img in classify_preds:
                for classify_pred_level_img in classify_pred_img:
                    classify_preds_temp.append(classify_pred_level_img.permute(1, 2, 0).reshape(-1, self.cate_out_channels))
            classify_preds = classify_preds_temp
            # classify_preds = [
            #             classify_pred_level_img.permute(1, 2, 0).reshape(-1, self.cate_out_channels)
            #         for classify_pred_level_img in classify_pred_img
            #     for classify_pred_img in classify_preds
            # ]
            flatten_classify_preds = torch.cat(classify_preds)
            num_ins = (flatten_classify_labels>0).sum().item()
            loss_classify = self.loss_cate(flatten_classify_preds, flatten_classify_labels, avg_factor=num_ins + 1)
        else:
            loss_classify = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)

        if self.enable_keypoints == True:
            keypoints_label_list, mask = multi_apply(
                self.parsing_keypoint_target_single,
                gt_keypoints_list,
                img_metas,
                mask_feat_size = mask_feat_size
            )
            keypoints_labels = torch.cat([heatmap_ins.unsqueeze(0) for heatmap_ins in keypoints_label_list])[mask]
            keypoints_preds = keypoints_preds[mask].permute(0,2,3,1).reshape(-1,20)

            loss_keypoints = self.loss_cate(keypoints_preds, keypoints_labels, avg_factor=(keypoints_labels>0).sum() + 1)
        else:
            loss_keypoints = torch.tensor(0,dtype=torch.float32, device=loss_ins.device)
        
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate,
            loss_center=loss_center,
            loss_offset=loss_offset,
            loss_seg=loss_seg,
            loss_class=loss_classify,
            loss_refine=loss_refine,
            loss_keypoints=loss_keypoints
            )

    def parsing_category_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               enable_heatmaploss,
                               mask_feat_size,
                               adapt_center):

        device = gt_labels_raw[0].device
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        # ins
        gt_parsings = []
        gt_labels_parsing = []
        gt_bboxes_parsing = []
        parsing_label_list = []
        ins_label_list = []
        cate_label_list = []
        classify_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        
        for i in range(self.cate_out_channels):
            gt_parsing = gt_parsing_raw == i+1
            if gt_parsing.max() == 0:
                continue
            else:
                try:
                    for ins in range(gt_parsing.shape[0]):
                        ys, xs = np.where(gt_parsing[ins] > 0)
                        if len(xs) == 0 or len(ys) == 0:
                            continue
                        else:
                            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                            gt_bboxes_parsing.append(np.array([x1,y1,x2,y2]))
                            gt_parsings.append(gt_parsing[ins])
                            gt_labels_parsing.append(i+1)
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
        gt_bboxes_parsing = torch.tensor(np.array(gt_bboxes_parsing)).float().to(device)
        gt_parsings = np.array(gt_parsings)
        gt_labels_parsing = torch.tensor(np.array(gt_labels_parsing)).to(device)
        
        if len(gt_bboxes_parsing) == 0:
            print("train a empty parsing image.")
            for num_grid in self.seg_num_grids:
                parsing_label = torch.tensor([], dtype=torch.int64, device=device)
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
                classify_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
                ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
                cate_label_list.append(cate_label)
                parsing_label_list.append(parsing_label)
                classify_label_list.append(classify_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
            return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, parsing_label_list

        gt_areas = torch.sqrt((gt_bboxes_parsing[:, 2] - gt_bboxes_parsing[:, 0]) * (
                gt_bboxes_parsing[:, 3] - gt_bboxes_parsing[:, 1]))
        
        
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)
            
            parsing_label = []
            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            classify_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                parsing_label = torch.tensor([], dtype=torch.int64, device=device)
                parsing_label_list.append(parsing_label)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label+58)
                classify_label_list.append(classify_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            
            gt_bboxes = gt_bboxes_parsing[hit_indices]
            gt_labels = gt_labels_parsing[hit_indices]
            #gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]
            gt_masks = gt_parsings[hit_indices.cpu().numpy(), ...].astype('uint8')
            
            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            try:
                center_ws, center_hs = center_of_mass(gt_masks_pt, adapt_center)
                valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                if enable_heatmaploss == True:
                    # cate_mask = F.interpolate(torch.from_numpy(seg_mask).float().unsqueeze(0).unsqueeze(0).to(device=device), size=(num_grid, num_grid), mode='bilinear').squeeze(0).squeeze(0)
                    cate_label = cate_label.float()
                    # cate_label[cate_label == 0] = 0.1
                    cate_label[top:(down+1), left:(right+1)] = gt_label
                else:
                    # if cate_label[top:(down+1), left:(right+1)].sum() > 0:
                    #     import pdb;pdb.set_trace()
                    cate_label[top:(down+1), left:(right+1)] = gt_label
                if self.enable_cate_decouple == True:
                    classify_label_temp = F.interpolate(torch.from_numpy(seg_mask).float().unsqueeze(0).unsqueeze(0).to(device=device)*gt_label, size=(num_grid, num_grid), mode='bilinear',align_corners=True).squeeze(0).squeeze(0).int().long()
                    classify_label[classify_label_temp>0] = gt_label
                seg_mask_tmp = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                if seg_mask_tmp.max()==0:
                    seg_mask = (mmcv.imrescale(seg_mask.astype(float), scale=1. / output_stride, interpolation='area')>0).astype('uint8')
                    seg_mask = torch.from_numpy(seg_mask).to(device=device)
                else:
                    try:
                        seg_mask = seg_mask_tmp
                        seg_mask = torch.from_numpy(seg_mask).to(device=device)
                    except Exception as e :
                        print(e)
                        import pdb; pdb.set_trace()
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        if cur_ins_label.max() == 0:
                            import pdb;pdb.set_trace()
                        ins_ind_label[label] = True
                        grid_order.append(label)
                        parsing_label.append(gt_label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)

            cate_label = cate_label - 1
            cate_label[cate_label<0] = self.cate_out_channels

            ins_label_list.append(ins_label)
            parsing_label_list.append(parsing_label)
            cate_label_list.append(cate_label)
            classify_label_list.append(classify_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)

            #import pdb;pdb.set_trace()
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, classify_label_list, parsing_label_list

    def parsing_center_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               enable_heatmaploss,
                               img_metas = None,
                               mask_feat_size = None):
        
        device = gt_labels_raw[0].device
        if self.enable_ori_grid:
            num_grid = self.seg_num_grids[0]
        else:
            num_grid = self.seg_num_grids[-2]
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        #import pdb;pdb.set_trace()
        #gt_masks_raw = F.interpolate(torch.from_numpy(gt_masks_raw).unsqueeze(0), size=upsampled_size, mode='bilinear').squeeze(0).numpy()
        # ins

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        gt_bboxes_parsing = []
        gt_masks = []
        grid_order_list = []
        

        for ins in range(gt_masks_raw.shape[0]):
            ys, xs = np.where(gt_masks_raw[ins] > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            else:
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                gt_bboxes_parsing.append(np.array([x1,y1,x2,y2]))
                gt_masks.append(gt_masks_raw[ins])
        
        if len(gt_bboxes_parsing) == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            ins_label_list.append(ins_label)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label_list.append(cate_label+1)
            return cate_label_list, ins_label_list
        
        gt_bboxes = np.array(gt_bboxes_parsing)
        gt_labels = gt_labels_raw
        #import pdb;pdb.set_trace()
        gt_masks = np.array(gt_masks)

        cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
        parsing_label = []
        ins_label = []
        grid_order = []
        ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

        half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
        half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

        # mass center
        gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
        center_ws, center_hs = center_of_mass(gt_masks_pt, self.enable_adapt_center)
        valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

        output_stride = 4

        for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
            if not valid_mask_flag:
                continue
            
            coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
            coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

            # left, top, right, down
            top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
            down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
            left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
            right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

            top = max(top_box, coord_h-1)
            down = min(down_box, coord_h+1)
            left = max(coord_w-1, left_box)
            right = min(right_box, coord_w+1)
            
            if enable_heatmaploss:
                cate_label = cate_label.float()
                #cate_mask = F.interpolate(torch.from_numpy(seg_mask).float().unsqueeze(0).unsqueeze(0).to(device=device), size=(num_grid, num_grid), mode='bilinear').squeeze(0).squeeze(0)
                #cate_label[cate_label == 0] = 0.1
                cate_label[top:(down+1), left:(right+1)] = gt_label
            else:
                cate_label[top:(down+1), left:(right+1)] = gt_label

            try:
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
            except Exception as e :
                print(e)
                import pdb; pdb.set_trace()
            for i in range(top, down+1):
                for j in range(left, right+1):
                    label = int(i * num_grid + j)

                    cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                device=device)
                    
                    cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                    
                    ins_label.append(cur_ins_label)
                    ins_ind_label[label] = True
                    grid_order.append(label)
        if len(ins_label) == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
        else:
            ins_label = torch.stack(ins_label, 0)
        
        cate_label = cate_label - 1
        cate_label[cate_label<0] = 1

        ins_label_list.append(ins_label)
        #parsing_label_list.append(parsing_label)
        ins_ind_label_list.append(ins_ind_label)
        grid_order_list.append(grid_order)
        cate_label_list.append(cate_label)
        
        return cate_label_list, ins_label_list, ins_ind_label_list, grid_order_list

    def parsing_keypoint_target_single(self,
                               gt_keypoints,
                               img_metas = None,
                               mask_feat_size = None):
        # gt_keypoints: (N*20*3)
        h, w, _ = img_metas['pad_shape']
        num_grid = self.seg_num_grids[0]
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        device = gt_keypoints.device

        if gt_keypoints.max() == -1:
            return torch.zeros((num_grid, num_grid), dtype=torch.int64, device=device), False
        
        N, C, P = gt_keypoints.shape

        gt_keypoints_list = torch.zeros((num_grid, num_grid), dtype=torch.int64, device=device)
        for ind in range(N):
            for key in range(C):
                if gt_keypoints[ind][key][0] > 0:
                    x0 = int((gt_keypoints[ind][key][0] / upsampled_size[1]) // (1. / num_grid))
                    y0 = int((gt_keypoints[ind][key][1] / upsampled_size[0]) // (1. / num_grid))
                    ul = int(np.floor(x0 - 1)), int(np.floor(y0 - 1))
                    br = int(np.ceil(x0 + 1)), int(np.ceil(y0 + 1))

                    cc, dd = max(0, ul[0]), min(br[0], num_grid)
                    aa, bb = max(0, ul[1]), min(br[1], num_grid)
                    gt_keypoints_list[aa:bb, cc:dd] = key+1

        return gt_keypoints_list, True

    def parsing_offset_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               img_metas = None,
                               mask_feat_size = None):

        device = gt_labels_raw[0].device
        num_grid = self.seg_num_grids[0]
        upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
        # ins
        gt_parsing_list = {}
        gt_masks = gt_masks_raw
        gt_masks_list = {}

        #gt_parsing_raw = F.interpolate(torch.from_numpy(gt_parsing_raw).unsqueeze(0), size=upsampled_size, mode='bilinear').squeeze(0).numpy()

        gt_bboxes_mask = []
        gt_masks = []
        gt_parsing = []

        for ins in range(gt_parsing_raw.shape[0]):
            ys, xs = np.where(gt_parsing_raw[ins] > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            else:
                #import pdb;pdb.set_trace()
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                gt_bboxes_mask.append(np.array([x1,y1,x2,y2]))
                gt_masks.append(gt_masks_raw[ins])
                gt_parsing.append(gt_parsing_raw[ins])
        
        if len(gt_bboxes_mask) == 0:
            offset_map = np.zeros((self.cate_out_channels, 2, num_grid, num_grid),
                              dtype=np.float32)
            weight_map = np.zeros((self.cate_out_channels, 2, num_grid, num_grid),
                              dtype=np.float32)
            return offset_map, weight_map

        gt_bboxes = np.array(gt_bboxes_mask)
        gt_masks = np.array(gt_masks)
        gt_parsing_raw = np.array(gt_parsing)

        # get center of masks and parsings
        for ins in range(gt_parsing_raw.shape[0]):
            gt_parsing_list[ins] = {}

            gt_mask = gt_parsing_raw[ins:ins+1] > 0
            gt_mask_pt = torch.from_numpy(gt_mask).to(device=device)
            #gt_mask_pt = F.interpolate(gt_mask_pt.unsqueeze(0).float(), size=upsampled_size, mode='bilinear').squeeze(0)

            for i in range(self.cate_out_channels):
                gt_parsing = gt_parsing_raw[ins:ins+1] == i+1
                if gt_parsing.max() == 0:
                    continue
                else:
                    gt_parsing_pt = torch.from_numpy(gt_parsing).to(device=device)
                    gt_parsing_pt = F.interpolate(gt_parsing_pt.unsqueeze(0).float(), size=upsampled_size, mode='bilinear',align_corners=True).squeeze(0)
                    center_ws, center_hs = center_of_mass(gt_parsing_pt, self.enable_adapt_center)
                    gt_parsing_list[ins][i] = (center_ws, center_hs)
            temp = []
            for k,v in gt_parsing_list[ins].items():
                temp.append(v)
            # center_ws, center_hs = torch.tensor(temp).mean(0)
            # gt_masks_list[ins] = (center_ws, center_hs)
            center_ws, center_hs = center_of_mass(gt_mask_pt, self.enable_adapt_center)
            gt_masks_list[ins] = (center_ws, center_hs)

        offset_map = np.zeros((self.cate_out_channels, 2, num_grid, num_grid),
                              dtype=np.float32)
        weight_map = np.zeros((self.cate_out_channels, 2, num_grid, num_grid),
                              dtype=np.float32)
        
        ws = (gt_bboxes[:, 2] - gt_bboxes[:, 0])
        hs = (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        half_ws = 0.5 * ws * self.sigma
        half_hs = 0.5 * hs * self.sigma
        #area = ws**2 + hs**2
        
        # mass center
        gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
        try:
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()

        output_stride = 4
        
        for ins in range(len(gt_parsing_list)):
            if not valid_mask_flags[ins]:
                continue
            gt_parsing_center = gt_parsing_list[ins]
            center_w, center_h = gt_masks_list[ins]
            try:
                half_h = half_hs[ins]
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            half_w = half_ws[ins]
            w = ws[ins]
            h = hs[ins]

            
            coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
            coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

            w = int((w / upsampled_size[1]) // (1. / num_grid))
            h = int((h / upsampled_size[0]) // (1. / num_grid))
            area = w**2 + h**2
            if area == 0:
                continue
            
            # left, top, right, down
            top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
            down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
            left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
            right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

            top = max(top_box, coord_h-1)
            down = min(down_box, coord_h+1)
            left = max(coord_w-1, left_box)
            right = min(right_box, coord_w+1)

            for idx in gt_parsing_center.keys():
                x, y = gt_parsing_center[idx]
                x = int((x / upsampled_size[1]) // (1. / num_grid))
                y = int((y / upsampled_size[0]) // (1. / num_grid))
                for pos_y in range(top, down+1):
                    for pos_x in range(left, right+1):
                        offset_x = pos_x - x
                        offset_y = pos_y - y
                        weight_map[idx, 0 ,pos_y, pos_x] = 1. / np.sqrt(area)
                        weight_map[idx, 1, pos_y, pos_x] = 1. / np.sqrt(area)
                        offset_map[idx, 0, pos_y, pos_x] = offset_x
                        offset_map[idx, 1, pos_y, pos_x] = offset_y

            #     filename = img_metas['filename'].split('/')[-1]
            #     img = torch.tensor(mmcv.imread(img_metas['filename']).transpose(2,0,1)).unsqueeze(0)
            #     img = F.interpolate(img.float(), size=(80,80), mode='bilinear').byte().squeeze(0).permute(1,2,0).numpy()
            #     img = img * 0.5
            #     img[y,x,:] = np.array((0,255,0),dtype='uint8')
            #     img[coord_h,coord_w,:] = np.array((255,255,255),dtype='uint8')
            #     cv.line(img,(x,y),(coord_w,coord_h),(255,255,0),5)
            #     mmcv.imwrite(img, "/home/notebook/code/personal/S9043252/multi-parsing/visual/offset/offset_map/gt/"+ str(idx) + "_" +filename)
                
            # import pdb;pdb.set_trace()

        return offset_map, weight_map

    def get_seg(self, cate_preds, kernel_preds, center_preds, offset_preds, segment_preds, classify_preds, keypoints_preds,
             seg_pred, img_metas, cfg, rescale=None, parsing_gt=None):
        if self.enable_cate_eval:
            self.get_seg_ori(cate_preds, kernel_preds, center_preds, offset_preds, seg_pred, img_metas, cfg)
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]
        
        # center_preds.shape: [imgs,S,S,1]
        # cate_preds.shape: [level, [imgs, S, S, 58]]
        # offset_preds.shape: [58 , [imgs, 2, S, S]]
        # kernel_preds.shape: [level, [imgs, 256, S, S]]
        #import pdb;pdb.set_trace()
        if not isinstance(img_metas, list):
            img_metas = img_metas.data[0]
        result_list = []
        for img_id in range(len(img_metas)):
            # process single images. 
            if self.enable_ori_grid == True:
                cate_pred_list = [
                    F.interpolate(cate_preds[i][img_id].detach().unsqueeze(0).permute(0,3,1,2), size=self.seg_num_grids[0], mode='bilinear',align_corners=True).permute(0,2,3,1) for i in range(num_levels)
                    #cate_preds[i][img_id].detach().unsqueeze(0) for i in range(num_levels)
                ]
                kernel_pred_list = [
                    F.interpolate(kernel_preds[i][img_id].detach().unsqueeze(0), size=self.seg_num_grids[0], mode='bilinear',align_corners=True).permute(0,2,3,1)
                    #kernel_preds[i][img_id].permute(1, 2, 0).detach().unsqueeze(0)
                                    for i in range(num_levels)
                ]
            else:
                cate_pred_list = [
                    cate_preds[i][img_id].detach().unsqueeze(0) for i in range(num_levels)
                ]
                kernel_pred_list = [
                    kernel_preds[i][img_id].permute(1, 2, 0).detach().unsqueeze(0)
                                    for i in range(num_levels)
                ]
            center_pred_list = center_preds[img_id].detach()

            offset_pred_list = [offset_preds[i][img_id].detach().unsqueeze(0) for i in range(self.cate_out_channels)]

            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)

            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)
            offset_pred_list = torch.cat(offset_pred_list, dim=0)
            result = self.get_parsing_single(cate_pred_list, seg_pred_list, kernel_pred_list, center_pred_list, offset_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, img_metas, rescale, parsing_gt)
            result_list.append(result)
        return result_list

    def get_parsing_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds, 
                       center_preds, 
                       offset_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       img_metas,
                       rescale=False, parsing_gt=None):

        assert len(cate_preds) == len(kernel_preds)
        
        import time
        t0 = time.time()
        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)
        cate_mask = np.zeros(ori_shape[:2], dtype=np.int8)

        center_gt = False
        if parsing_gt and center_gt:
            #import pdb;pdb.set_trace()
            parsing_gt = parsing_gt[0].cpu()
            center_label_list, _, _, _ = self.parsing_center_target_single(None, torch.tensor([1]*parsing_gt[0].shape[0]).cuda(), parsing_gt[0].numpy(), parsing_gt[0].numpy(),False,img_metas,seg_preds.size()[-2:])
            if self.enable_ori_grid == True:
                gt_center_map = F.interpolate((center_label_list[0] == (1)).detach().unsqueeze(0).unsqueeze(0).float(), size=self.seg_num_grids[0], mode='bilinear',align_corners=True).squeeze(0)
                center_preds = gt_center_map.permute(1, 2, 0)
            else:
                for p_i in range(self.cate_out_channels):
                    gt_cate_maps_level = []
                    for level_i in range(5):
                        gt_cate_maps_level.append((cate_label_list[level_i] == (p_i+1)).float().unsqueeze(0))
                    gt_cate_maps_level = torch.cat(gt_cate_maps_level).unsqueeze(0)
                    gt_cate_maps.append(gt_cate_maps_level)
                gt_cate_maps = torch.cat(gt_cate_maps)
                cate_maps = gt_cate_maps.cuda()
                cate_preds = cate_maps
        elif parsing_gt:
            parsing_gt = parsing_gt[0].cpu()
        
        # import pdb;pdb.set_trace()
        # Screen the center of human.
        # center_preds.shape: [S,S,1]
        center_preds = center_preds.unsqueeze(0).permute(0,3,1,2)
        center_nms = nn.functional.max_pool2d(center_preds, (5,5), stride=1, padding=2)
        center_nms = torch.eq(center_nms, center_preds)
        center_preds = (center_preds*center_nms).squeeze(0).permute(1,2,0)

        # TODO: 添加超参数
        inds = (center_preds > cfg.ctr_score)
        center_scores = center_preds[inds]
        if len(center_scores) == 0:
            filename = img_metas[0]['filename'].split('/')[-1]
            print("Can't detect any parsing of " + filename + " because of center.")
            mmcv.imwrite(cate_mask, "./eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
            return None
        
        # import pdb;pdb.set_trace()
        # Process each person
        # offset_preds.shape: [58 , [2, S, S]]
        # seg_preds.shape: [1, 256, H, W]
        ys, xs = torch.where(inds.squeeze(2))
        offset_preds = offset_preds.permute(2, 3, 0, 1)
        # cate_preds.shape: [5, 80, 80, 58]
        # kernel_preds.shape: [5, 80, 80, 256]
        cate_preds = cate_preds.permute(1,2,3,0)
        kernel_preds = kernel_preds.permute(1,2,0,3)
        
        ind_sort = torch.argsort(center_scores, descending=True)
        center_scores = center_scores[ind_sort]
        xs, ys = xs[ind_sort], ys[ind_sort]
        final_scores = []

        instance_kernel_list = []
        cate_maps = cate_preds.permute(2,3,0,1)

        if parsing_gt is not None:
            _, cate_label_list, _, _, _,_ = self.parsing_category_target_single(None,parsing_gt, None, parsing_gt[0].numpy(),False,seg_preds.size()[-2:])
            gt_cate_maps = []
            if self.enable_ori_grid == True:
                for p_i in range(self.cate_out_channels):
                    gt_cate_maps_level = []
                    for level_i in range(5):
                        gt_cate_maps_level.append(F.interpolate((cate_label_list[level_i] == (p_i+1)).detach().unsqueeze(0).unsqueeze(0).float(), size=self.seg_num_grids[0], mode='bilinear',align_corners=True).squeeze(0))
                    gt_cate_maps_level = torch.cat(gt_cate_maps_level).unsqueeze(0)
                    gt_cate_maps.append(gt_cate_maps_level)
            else:
                for p_i in range(self.cate_out_channels):
                    gt_cate_maps_level = []
                    for level_i in range(5):
                        gt_cate_maps_level.append((cate_label_list[level_i] == (p_i+1)).float().unsqueeze(0))
                    gt_cate_maps_level = torch.cat(gt_cate_maps_level).unsqueeze(0)
                    gt_cate_maps.append(gt_cate_maps_level)
            gt_cate_maps = torch.cat(gt_cate_maps)
            cate_maps = gt_cate_maps.cuda()
            cate_preds = cate_maps
            #import pdb;pdb.set_trace()
            
        
        # import pdb;pdb.set_trace()
        mp = torch.nn.MaxPool2d(1, 1, 0)
        maxm = torch.eq(mp(cate_maps), cate_maps)
        maxm = cate_maps * maxm
        maxm = maxm.permute(2, 3, 0, 1)
        cate_maps = mp(cate_maps).permute(2, 3, 0, 1)
        
        seg_masks = []
        parsing_preds = []
        offset_vis = []

        #print("center: ", time.time()-t0)
        for ind in range(len(xs)):
            #print("xs_",ind,": ", time.time()-t0)
            
            x, y = xs[ind], ys[ind]
            offset = offset_preds[y][x]
            part_center = torch.tensor([x,y], device=x.device) - offset
            left,top = part_center.min(0)[0]
            right,bottom = part_center.max(0)[0]
            sqr = (torch.sqrt((right-left)*(bottom-top))*0.5).int()

            kernel_list = []
            label_list = []
            score_list = []
            offset_list = []
            level_list = []
            s_max_id = self.seg_num_grids[0]-1
            part_center = torch.clamp(part_center, 0, s_max_id).int()

            maxm_pi = maxm > cfg.score_thr
            maxm_si = maxm[maxm_pi]
            if len(maxm_si) == 0:
                continue
            max_ys, max_xs, p_i, level_is = torch.where(maxm_pi)
            x_p, y_p = part_center[p_i][:,0], part_center[p_i][:,1]
            d_x, d_y = max_xs - x_p, max_ys - y_p
            dist = abs(d_x) + abs(d_y)
            d_weight = torch.exp(- ((dist) ** 2) / ( sqr ** 2).float())
            dist_score = d_weight * maxm_si
            coord = []
            max_inds = []
            #print("start of for",": ", time.time()-t0)
            one_hot = torch.nn.functional.one_hot(p_i,num_classes=self.cate_out_channels).bool()
            expand_dist = dist_score.unsqueeze(1).expand(dist_score.shape[0],self.cate_out_channels)
            try:
                max_inds = torch.argmax(expand_dist * one_hot,0)
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            inds = dist_score[max_inds]>cfg.cate_score_thr
            if len(inds) == 0:
                continue
            max_inds = max_inds[inds]
            score_list = dist_score[max_inds]
            label_list = p_i[max_inds]
            coord = (max_ys[max_inds], max_xs[max_inds],level_is[max_inds])
            
            #print("end of for",": ", time.time()-t0)
            try:
                kernel_list = kernel_preds[coord[0],coord[1],coord[2],:]
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()

            offset_list = (label_list, x, y, coord, score_list)
            offset_vis.append(offset_list)
            if len(kernel_list) == 0:
                #print(x," ",y," ",center_scores[ind])
                continue
            
            final_scores.append(center_scores[ind])

            level_list = coord[2]
            inds = torch.argsort(score_list)
            
            label_list = label_list[inds]
            score_list = score_list[inds]
            kernel_list = kernel_list[inds]    

            #print("start of conv",": ", time.time()-t0)    

            I, N = kernel_list.shape
            kernel_list = kernel_list.view(I, N, 1, 1)
            parsing_pred = F.conv2d(seg_preds, kernel_list, stride=1).squeeze(0).sigmoid()
            ins_mask = parsing_pred > cfg.mask_thr
            sum_masks = ins_mask.sum((1, 2)).float()
            strides = [ [8, 8, 16, 32, 32][level.item()] for level in level_list]
            #strides = (level_list+1) * 8
            #print("end of conv",": ", time.time()-t0) 

            if self.enable_moi:
                mois = []
                moi_feats = []
                moi_feats_tmp = []
                memory_limit = 50
                for ins_idx in range(len(ins_mask)):
                    try:
                        ys, xs = torch.where(ins_mask[ins_idx])
                        l,t = xs.min(), ys.min()
                        r,b = xs.max(), ys.max()
                        mois.append(torch.tensor([l,t,r,b], dtype=torch.float32,device=t.device).unsqueeze(0))
                        parsing_feat = ins_mask[ins_idx]
                        base_feat = seg_preds
                        conv_feat = parsing_pred[ins_idx]
                        moi_feat = torch.cat([parsing_feat.float().unsqueeze(0), base_feat.squeeze(0), conv_feat.unsqueeze(0)])
                        moi_feats_tmp.append(moi_feat.unsqueeze(0))
                    except Exception as e:
                        print(e)
                        import pdb;pdb.set_trace()
                    if memory_limit != 0 and ins_idx != (len(ins_mask)-1):
                        memory_limit -= 1
                    else:
                        moi_feats.append(self.roi_extractor(torch.cat(moi_feats_tmp),mois))
                        moi_feats_tmp = []
                        mois = []
                        memory_limit = 50
                moi_feats = torch.cat(moi_feats).detach()
                moi_feats = self.class_conv1(moi_feats)
                moi_feats = self.class_conv2(moi_feats).view(moi_feats.shape[0],-1)
                
                moi_feats = self.relu(self.class_fc1(moi_feats))
                moi_feats = self.relu(self.class_fc2(moi_feats))
                
                moi_class = torch.argmax(moi_feats)

            keep = sum_masks.cpu() > torch.tensor(strides)
            ins_mask = ins_mask[keep]
            parsing_pred = parsing_pred[keep]
            label_list = label_list[keep]
            score_list = score_list[keep]
            sum_masks = sum_masks[keep]
            if self.enable_moi:
                import pdb;pdb.set_trace()
                moi_class = moi_class[keep]
            #import pdb;pdb.set_trace()
            # maskness.
            parsing_scores = (parsing_pred * ins_mask.float()).sum((1, 2)) / sum_masks
            score_list *= parsing_scores

            # add a NMS
            cate_labels = label_list
            cate_scores = score_list
            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > cfg.nms_pre:
                sort_inds = sort_inds[:cfg.nms_pre]
            ins_mask = ins_mask[sort_inds, :, :]
            parsing_pred = parsing_pred[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]
            if self.enable_moi:
                moi_class = moi_class[sort_inds]
            #print("start of NMS",": ", time.time()-t0) 
            # Matrix NMS
            cate_scores, _, ins_mask, keep_inds = matrix_nms(ins_mask, torch.tensor([1 for i in cate_labels],device=ins_mask.device), cate_scores.cuda(),
                                        kernel=cfg.kernel,sigma=cfg.sigma, mask_area=sum_masks)
            # cate_scores, _, ins_mask, keep_inds = matrix_nms(ins_mask, cate_labels.cuda(), cate_scores.cuda(),
            #                             kernel=cfg.kernel,sigma=cfg.sigma, mask_area=sum_masks)
            #print("end of NMS",": ", time.time()-t0) 
            parsing_pred = parsing_pred[keep_inds]
            cate_labels = cate_labels[keep_inds]
            if self.enable_moi:
                moi_class = moi_class[keep_inds]
            #import pdb;pdb.set_trace()
            keep = cate_scores >= cfg.cate_update_thr
            if keep.sum() == 0:
                filename = img_metas[0]['filename'].split('/')[-1]
                print("Can't detect any parsing of one person in" + filename + " because of NMS.")
                if not self.enable_cate_eval:
                    mmcv.imwrite(cate_mask, "./eval_file/pred_cate"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
                continue
            ins_mask = ins_mask[keep, :, :]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]
            parsing_pred = parsing_pred[keep]
            if self.enable_moi:
                moi_class = moi_class[keep]
            
            # sort and keep top_k
            sort_inds = torch.argsort(cate_scores, descending=False)
            if len(sort_inds) > cfg.max_per_img:
                sort_inds = sort_inds[:cfg.max_per_img]
            ins_mask = ins_mask[sort_inds, :, :]
            #seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]
            parsing_pred = parsing_pred[sort_inds]
            #print("start of interpolate",": ", time.time()-t0) 
            if self.enable_moi:
                moi_class = moi_class[sort_inds]
            parsing_pred = F.interpolate(parsing_pred.unsqueeze(0), size=upsampled_size_out, mode='bilinear',align_corners=True)[:,:,:h,:w]
            parsing_pred = F.interpolate(parsing_pred, size=ori_shape[:2], mode='bilinear',align_corners=True).squeeze(0)
            ins_mask = parsing_pred > cfg.mask_thr
            ins_masks = torch.zeros(ins_mask.shape[-2:], dtype=torch.uint8, device=ins_mask.device)
            ins_preds = torch.zeros(ins_mask.shape[-2:], dtype=torch.float32, device=ins_mask.device)
            #print("end of interpolate",": ", time.time()-t0)
            #ins_mask = ins_mask.byte() * (cate_labels+1).unsqueeze(1).unsqueeze(1).cuda()
            for p_i in range(len(cate_labels)):
                if self.enable_moi:
                    ins_masks[ins_mask[p_i]] = (ins_mask[p_i].byte() * (moi_class[p_i]+1))[ins_mask[p_i]]
                else:
                    ins_masks[ins_mask[p_i]] = (ins_mask[p_i].byte() * (cate_labels[p_i]+1))[ins_mask[p_i]]
                #ins_preds[ins_mask[p_i]] = parsing_pred[p_i][ins_mask[p_i]]
            #print("end of add",": ", time.time()-t0) 
            seg_masks.append(ins_masks.cpu().detach().numpy())
            #parsing_preds.append(ins_preds.unsqueeze(0))
            #print("end",": ", time.time()-t0) 
            #import pdb;pdb.set_trace()
        
        if len(seg_masks) == 0:
            filename = img_metas[0]['filename'].split('/')[-1]
            print("Can't detect any parsing of " + filename + " because of seg_mask.")
            if not self.enable_cate_eval:
                mmcv.imwrite(cate_mask, "./eval_file/pred_cate"+img_metas[0]['filename'].split('/')[-1].split('/')[-1].split('.')[0] + '.png')
            return None
        
        # # human matrix nms
        
        # human_mask = [torch.from_numpy(seg_mask).unsqueeze(0) for seg_mask in seg_masks]
        # human_mask = torch.cat(human_mask).cuda()
        # parsing_preds = torch.cat(parsing_preds)
        # num_h, h, w = parsing_preds.shape
        # human_scores = []
        # for id_h in range(num_h):
        #     parsing_ind = parsing_preds[id_h]>0
        #     human_scores.append((parsing_preds[id_h][parsing_ind].mean())* center_scores[id_h])
        
        # sum_masks = human_mask.sum((1, 2)).float()
        # human_scores, _, human_mask, keep_inds = matrix_nms(human_mask, torch.tensor([1 for i in human_scores],device=human_mask.device), torch.stack(human_scores),
        #                                 kernel=cfg.kernel,sigma=cfg.sigma, mask_area=sum_masks)
        
        # keep = human_scores > cfg.update_thr
        # human_mask = human_mask[keep]
        # #import pdb;pdb.set_trace()
        # human_scores = human_scores[keep]
        # seg_masks = human_mask.cpu().numpy()
        # human_masks =[]
        # final_scores = human_scores
        
        if not self.enable_cate_eval:
            for i in range(len(seg_masks)):
                seg_mask_ind = seg_masks[i]>0
                cate_mask[seg_mask_ind] = seg_masks[i][seg_mask_ind]
            mmcv.imwrite(cate_mask, "eval_file/pred_cate/"+img_metas[0]['filename'].split('/')[-1].split('.')[0] + '.png')
        
        cate_labels = None
        
        return seg_masks, offset_vis, final_scores