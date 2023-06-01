import torch
import torch.nn as nn

from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
import copy


@DETECTORS.register_module
class SingleStageInsParsingDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageInsParsingDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = builder.build_head(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                    f'be implemented in {self.__class__.__name__}'

        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x             #x为一组张量，长度为5

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_parsing=None,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        gt_semantic_seg=None,
                        gt_keypoints=None):
        
        x = self.extract_feat(img)  #数据的输入
        batch_inputs= (gt_bboxes, gt_labels, gt_masks, gt_parsing, gt_semantic_seg, gt_keypoints, img_metas, self.train_cfg)

        outs = self.mask_head(x, batch_inputs)   #调用相关模型，在parsing_offset_v1_basic里面
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_parsing, gt_semantic_seg, gt_keypoints, img_metas, self.train_cfg)
        losses = self.mask_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False, **kwargs):
        x = self.extract_feat(img)
        outs = self.mask_head(x, eval=True) 
        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        seg_result = self.mask_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
