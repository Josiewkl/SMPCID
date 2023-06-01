import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

BN_MOMENTUM = 0.1 #用于权值优化
logger = logging.getLogger(__name__)

def build_backbone(cfg, is_train):
    model = HRNet(cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model

def conv3x3(in_planes, out_planes, stride=1, dilation=1): #返回一个核为3*3，步长为1，padding为1的2d卷积层
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class BasicBlock(nn.Module):  #定义基础残差模块，卷积核都是2个3x3
    expansion = 1 #输入和输出的通道数是一致的

    #pdb.set_trace()
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        residual = x
        # pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # pdb.set_trace()
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    #pdb.set_trace()


class Bottleneck(nn.Module):  #1x1，3x3，1x1共三个卷积核组成
    expansion = 4 #输出通道数是输入通道数的4倍

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        #输入维度为planes，输出维度为planes*4
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        # num_branches: 当前stage分支的平行子网络数目
        # blocks: BasicBlock或者Bottleneck
        # num_blocks: BasicBlock或者Bottleneck数目
        # num_inchannels: 输入通道数
        # stage=2时, num_inchannels=[32,64]
        # stage=3时, num_inchannels=[32,64,128]
        # stage=4时, num_inchannels=[32,64,128]
        # num_channels: 输出通道数
        # stage=2时, num_channels=[32,64]
        # stage=3时, num_channels=[32,64,128]
        # stage=4时, num_channels=[32,64,128,256]
        # fuse_method: 融合方式，默认为fuse
        # multi_scale_output: HRNet-W32配置下，所有stage均为multi_scale_output=True
        super(HRModule, self).__init__()
        #检测输入参数
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        #为每个分支构建分支网络
        #stage=2，3，4，平行子网分支数num_branches=2,3,4
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        #创建多尺度融合层
        # stage=2,3,4时，len(self.fuse_layers)分别为2,3,4，与其num_branches在每个stage的数目是一致的
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, #检查分支数目是否与其他参数列表中的元素数目相等
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None #下采样层
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        #函数通过实例化块来添加更多的层
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers) #返回一个包含所有层的 nn.Sequential 实例

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = [] #定义一个空列表

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)) #构建单个分支网络层，并添加到列表中

        return nn.ModuleList(branches) #将所有的分支网络层打包成一个模块列表，并将其返回。

    def _make_fuse_layers(self): #进行多个分辨率层的融合
        if self.num_branches == 1:
            return None

        #平行子网络分支数
        num_branches = self.num_branches
        #输入通道数
        num_inchannels = self.num_inchannels
        fuse_layers = []
        #为每个分支都创建对应的特征融合网络,如果multi_scale_output=1,则只需要一个特征融合网络
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                ##多种情况的讨论
                # 1、当前分支信息传递到上一分支(沿论文图示scale方向)的下一层(沿论文图示depth方向),进行上采样,分辨率加倍
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                #2、当前分支信息传递到当前分支(论文图示沿scale方向)的下一层(沿论文图示depth方向),不做任何操作,分辨率相同
                elif j == i:
                    fuse_layer.append(None)
                #3、当前分支信息传递到当前分支(论文图示沿scale方向)的下一层(沿论文图示depth方向),分辨率减半,分辨率减半
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers) #返回一个对象，其中包含所有分支对的融合层

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:#判断当前网络是否只有一个分支，y则进行前向传播，n则对所有分支进行遍历并对每个分支进行前向传播
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = [] #定义空列表，用于存储所有分支的输出结果

        #函数通过循环遍历所有的融合层，对输入数据进行融合操作
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}

class HRNet(nn.Module):
    def __init__(self, cfg):
        super(HRNet, self).__init__()
        # stem net 获取原始特征图N11

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False) #卷积层
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM) #BN层
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True) #激活函数
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.BACKBONE
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES #层数
        num_channels_last = [256] #通道数目
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i+1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i+2), stage)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            ) #Sequential将两个层叠在一起

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer) #当前层特征图中通道数
        num_branches_pre = len(num_channels_pre_layer) #前一层特征图中通道数

        transition_layers = []
        for i in range(num_branches_cur): 
            if i < num_branches_pre: #当前分支在前一层中有对应的分支
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]: #判断当前层中的通道数是否与前一层中的通道数不同
                    transition_layers.append(nn.Sequential( #该过渡层由 3x3 卷积层组成，然后是批量归一化和 ReLU 激活
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None) #通道数相同
            else: #当前分支在前一层中没有相应的分支
                conv3x3s = [] # 3x3 卷积层以对特征图进行下采样
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                     multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x) #数据传输，各层连接关系

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i+1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i+2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)

        x = torch.cat([y_list[0], \
                       F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False), \
                       F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False), \
                       F.interpolate(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)], 1)

        return x

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules(): #对模型中的Conv2d和BatchNorm2d进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001) #权重使用标准差为0.001的正态分布进行初始化，偏差初始化为0
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #权重初始化为1，偏差初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules(): #权重和偏差进行初始化，均设置为0
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        #根据预训练模型的路径加载预训练权重，并根据预训练权重中的权重名称，对当前模型中的权重进行初始化
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False) #strict参数为False，表示预训练模型中的权重可以不完全匹配当前模型的权重，只需要匹配到的部分进行加载。