import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose

from .mhp_data import get_data
import json
from tqdm import tqdm
from mmdet.core import parsing_iou, eval_parsing_ap

import torch
import cv2

@DATASETS.register_module
class MHP(Dataset):
    """MHP dataset for parsing.

    Annotation format:
    [
        {
            
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self._set = 'train' if test_mode == False else 'val'
        self.ann_file = ann_file
        # join paths if data_root is specified
        if self.data_root is not None: #如果 data_root 路径不是 None，表示数据路径是绝对路径，则执行以下步骤，将输入的相对路径转换为绝对路径
            if not osp.isabs(self.ann_file): #如果 'ann_file' 对应的路径是相对路径而不是绝对路径，将 'ann_file' 路径和 'data_root' 路径拼接成绝对路径
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)): #如果 img_prefix 路径不是 None 且为相对路径，则将其设置为与 data_root 结合的绝对路径
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = json.load(open("cache/dat_list_{}.json".format(self._set), 'r')) #加载图像信息，将 dat_list_{}.json 文件中的图像信息读取并赋值给 img_infos 变量。其中，self._set 表示数据集的名称。
        self.ann_infos = json.load(open("cache/data_list_{}.json".format(self._set), 'r')) #加载标注信息，将 data_list_{}.json 文件中的标注信息读取并赋值给 ann_infos 变量。其中，self._set 表示数据集的名称。
        for i, img_info in enumerate(self.img_infos): #遍历 img_infos 列表，使用 enumerate() 函数获取 i 和 img_info 的值
            img_info['filepath'] = data_root + img_info['filepath'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
        for i, ann_info in enumerate(self.ann_infos):
            ann_info['filepath'] = data_root + ann_info['filepath'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
            for bbox in ann_info['bboxes']:
                bbox['ann_path'] = data_root + bbox['ann_path'].split('/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]
        
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file) #根据给定的proposal_file加载预先生成的候选框
        else:
            self.proposals = None
        # filter images too small
        if not test_mode: #如果不处于测试模式
            valid_inds = self._filter_imgs() #将数据集中尺寸过小的图像过滤，并返回合法的图像索引列表 valid_inds
            self.img_infos = [self.img_infos[i] for i in valid_inds] #将 img_infos 列表中没有通过过滤器的图像过滤掉，仅保留合法的图像。
            if self.proposals is not None: #通过索引列表过滤掉与 img_infos 不匹配的提案数据，仅保留与图像对应的提案，并重新赋值给 self.proposals
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag() #设置数据集的分组信息
        # processing pipeline
        self.pipeline = Compose(pipeline) #用于构建一个数据处理 pipeline，Compose 函数将给定的多个处理步骤组合成一个新的操作步骤，最终得到经过所有处理函数处理后的数据

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file): #从指定的 COCO 格式的标注文件（例如 JSON 文件）中读取标注数据
        return mmcv.load(ann_file) #调用 mmcv.load() 函数进行数据的读取

    def load_proposals(self, proposal_file): #用于从给定的文件中读取 proposals（候选框）数据
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx): #于返回数据集中第 idx 张图像的注释信息
        return self.img_infos[idx]['ann'] #'ann' 表示图像的注释信息，包括图像的标签，框的坐标和面积等

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix #数据集路径信息中的 img_prefix 字段存储到 results 中
        results['seg_prefix'] = self.seg_prefix 
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['parsing_fields'] = []
        results['keypoints_fields'] = []

    def _filter_imgs(self, min_size=32): #用于过滤尺寸太小的图像
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self): #用于为数据集中的图像分组
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8) #self.flag 是一个数组，用于存储每张图像所属的分组信息，0 表示属于第一组，1 表示属于第二组
        for i in range(len(self)): #遍历数据集中的每张图像，如果该图像的宽高比大于 1，则将其标记为第二组（self.flag[i] = 1），否则标记为第一组（self.flag[i] = 0）
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx): #用于在数据集中随机选择另一张分组相同的图像，用于在相同的分组（如 S0/S1）中随机选择另一张图像（与输入的 idx 不同）来构成对比学习任务中的负样本
        pool = np.where(self.flag == self.flag[idx])[0] #用于在数据集中筛选与目标图像分组相同的所有图像，并生成一个索引列表 pool ，self.flag 表示每张图像所属的分组，0 和 1 分别表示两个不同的分组。
        return np.random.choice(pool) #随机选择 pool 中的一个索引作为匹配的图像

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            #import pdb;pdb.set_trace()
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data #返回的是一个样本的特征、标注信息及相应的负样本信息，用于训练对比学习模型。

    def prepare_train_img(self, idx):
        """ 
        return
        result :
            {'img_info' :{
                    'filepath' : img_add
                    'width' : image.size[1]
                    'height' : image.size[0]
                    'bboxes' : [{
                                        'class': 'person',
                                        'ann_path': ann_root + ann_add,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2
                                        }]
                }
            'anno_info': {
                    'filepath' : img_add
                    'width' : image.size[1]
                    'height' : image.size[0]
                    'bboxes' : [{
                                        'class': 'person',
                                        'ann_path': ann_root + ann_add,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2
                                        }]
                }
            'img_prefix' = self.img_prefix
            'seg_prefix' = self.seg_prefix
            'proposal_file' = self.proposal_file
            'bbox_fields' = []
            'mask_fields' = []
            'seg_fields' = []
            'parsing_fields' = []
        """
        results = {}
        
                      
        img_info = self.img_infos[idx] #获取图像信息
        ann_info = self.ann_infos[idx] #获取标注信息
        anno = [obj for obj in ann_info] #将标注信息存储在列表anno中
        m = np.zeros((img_info['height'], img_info['width'])) #创建一个大小为(img_info['height'], img_info['width']) 的纯0数组m，并遍历标注信息
        
        if self.dataset == 'coco':
            import pycocotools #使用pycocotools中的函数frPyObjects()和decode()计算并更新m中的值，以创建一个掩码。
            for obj in anno:
                if obj['iscrowd']:
                    rle = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    m += pycocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)

        mask = m < 0.5 #然后根据这个掩码的阈值，将掩码转变为二进制像素掩码。

        #将排除掉iscrowd参数为0的标注信息，并将num_keypoints参数大于零的标注信息存储在列表anno中
        anno = [obj for obj in anno if obj['iscrowd'] == 0 and obj['num_keypoints'] > 0] 
        num_people = len(anno)  #计算人数
        area = np.zeros((num_people, 1)) #确定面积
        bboxs = np.zeros((num_people, 4, 2))  #确定边界框
        keypoints = np.zeros((num_people, self.num_keypoints, 3)) #确定关键点
        centers = np.zeros((num_people, 1, 3)) #确定中心坐标

        for i, obj in enumerate(anno):
                keypoints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])
                area[i, 0] = obj['bbox'][2] * obj['bbox'][3]   #area 是一个二维的 numpy 数组（第一维度是人数，第二维度是1），它用于存储每个人体边界框的面积。
                #obj['bbox'] 是一个包含4个元素的数组，它保存了 bounding box 左上角点 (x,y) 和边界框宽度 w 和高度 h
                #obj['bbox'][2] * obj['bbox'][3] 对应的就是边界框的面积（即边界框宽度与高度的乘积）
                #将当前标注信息的面积值存储在 area数组中的第 i 行
                bboxs[i, :, 0], bboxs[i, :, 1] = obj['bbox'][0], obj['bbox'][1]
                #bboxs 是一个三维 numpy 数组，它的第一个维度对应人数，第二个维度对应边界框角点数（这里是4个角点），第三个维度包含x和y位置
                #bboxs[i, :, 0] 和 bboxs[i, :, 1] 分别用于存储边界框的四个角点的x坐标和y坐标
                #左上角点的x坐标 obj['bbox'][0] 存储在 bboxs[i, :, 0] 的第0列中，左上角点的y坐标 obj['bbox'][1] 存储在 bboxs[i, :, 1] 的第0列中
                bboxs[i, 1, 0] += obj['bbox'][2] #将当前的右上角点向右偏移 obj['bbox'][2] 个单位，从而得到边界框的右上角点的完整坐标
                bboxs[i, 2, 1] += obj['bbox'][3] #将当前的左下角点向下偏移 obj['bbox'][3] 个单位，从而得到边界框的左下角点的完整坐标
                bboxs[i, 3, 0] += obj['bbox'][2]; bboxs[i, 3, 1] += obj['bbox'][3] #将当前的右下角点向右偏移 obj['bbox'][2] 个单位(x坐标)，将当前的右下角点向下偏移 obj['bbox'][3] 个单位（y坐标），从而得到边界框的右下角点的完整坐标
                #bboxs[i, 1, 0]、bboxs[i, 2, 1]、bboxs[i, 3, 0]、bboxs[i, 3, 1] 分别用于存储边界框的右上角、左下角和右下角的坐标
                #obj['bbox'][2] 和 obj['bbox'][3] 分别代表边界框的宽度和高度


        if self.transform:
            img, mask, keypoints, area, bboxs = self.transform(img, mask, keypoints, area, bboxs)

        for i, obj in enumerate(anno):   #将图像中的关键点信息进行处理，并得出每个实例的中心坐标
            if not self.dataset == 'crowdpose':
                if area[i, 0] < 32 ** 2:  #如果实例的面积小于 32 * 32，则将其中心点标记为0并跳过后续的操作，否则开始计算实例的中心坐标
                    centers[i, :, 2] = 0
                    continue
            vis = (keypoints[i, :, 2:3] > 0).astype(np.float32) #提取关键点可见性信息：每个关键点的可见性由 keypoints 中的第三列表示，将其转换为 float 类型存储在 vis 中（可见性为 1.0，不可见性为 0.0)
            keypoints_sum = np.sum(keypoints[i, :, :2] * vis, axis=0) #所有可见关键点坐标的加权和;keypoints[i, :, :2]是第i个人的所有关键点坐标，
             #vis是一个与之同形状的矩阵，其中每个元素代表相应关键点是否可见，如果可见则为1，否则为0；将两个矩阵相乘，并在第2个维度上求和
            num_vis_keypoints = len(np.nonzero(keypoints[i, :, 2])[0]) #算可见关键点的数量
            if num_vis_keypoints <= 0: centers[i, 0, 2] = 0; continue #如果没有可见关键点，则将中心点标记为0并跳过后续的操作。
            centers[i, 0, :2] = keypoints_sum / num_vis_keypoints #计算关键点的中心点坐标。
            centers[i, 0, 2] = 2 #将中心点坐标的第三个元素标记为 2

        keypoints_with_centers = np.concatenate((keypoints, centers), axis=1) #将中心点坐标 centers 与 keypoint 关键点坐标沿着第二个维度进行拼，#形成一个新的三维数组；
        #第一维代表实例编号，第二维代表关键点+中心点坐标，第三维代表坐标轴

        heatmap_with_centers, _ = self.heatmap_generator(keypoints_with_centers, bboxs) #生成人体关键点热力图和人体中心点坐标；_ 是一个没有任何实际作用的变量，只是为了占位符而已

        inst_coords, inst_heatmaps, inst_masks = self.get_inst_annos(centers, keypoints, area, bboxs) #计算实例的坐标、heatmap 和掩膜
        if len(inst_coords) > 0:
            inst_coords = np.concatenate(inst_coords, axis=0) #将实例的坐标数据拼接成一个数组
            inst_heatmaps = np.concatenate(inst_heatmaps, axis=0)
            inst_masks = np.concatenate(inst_masks, axis=0)
            results['instance_coord'] = torch.from_numpy(inst_coords) #将处理完的实例坐标数据存储在字典 results 中的键 'instance_coord' 中，并转换为 PyTorch 的张量类型
            results['instance_heatmap'] = torch.from_numpy(inst_heatmaps)
            results['instance_mask'] = torch.from_numpy(inst_masks)
        results['image'] = img
        results['multi_heatmap'] = torch.from_numpy(heatmap_with_centers)
        results['multi_mask'] = torch.from_numpy(mask[None, :, :])
   

        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results) ##调用 pre_pipeline 方法对 results 进行预处理
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.ann_infos[idx]
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None: #如果提供了候选框信息，则将其加入 results 字典中
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results) #调用 pre_pipeline 方法对 results 进行预处理
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None): #用于对模型在验证集上的性能进行评估
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        dat_list = get_data(self.data_root, 'val')
        _iou, _miou, mean_acc, pixel_acc = parsing_iou('work_dirs/pred_cate', 'work_dirs/gt_cate', 59) #计算模型在验证集上的 Segmentation IoU、Pixel Accuracy 和 Mean Accuracy
        parsing_result = {'mIoU': _miou, 'pixel_acc': pixel_acc, 'mean_acc': mean_acc}

        for i, iou in enumerate(_iou):
            print(' {:<30}:  {:.2f}'.format(parsing_name[i], 100 * iou))

        print('----------------------------------------')
        print(' {:<30}:  {:.2f}'.format('mean IoU', 100 * _miou))
        print(' {:<30}:  {:.2f}'.format('pixel acc', 100 * pixel_acc))
        print(' {:<30}:  {:.2f}'.format('mean acc', 100 * mean_acc))

        all_ap_p, all_pcp, miou = eval_parsing_ap(results, dat_list, nb_class=59, ovthresh_seg=0.5, From_pkl=False, Sparse=False) #调用 eval_parsing_ap 方法计算 AP、PCP 和 Mean IoU
        ap_p_vol = np.mean(all_ap_p)

        print('~~~~ Summary metrics ~~~~')
        print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(ap_p_vol))
        print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(all_ap_p[0]))
        print(' Average Precision based on part (APp)               @[mIoU=0.20      ] = {:.3f}'.format(all_ap_p[1]))
        print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(all_ap_p[2]))
        print(' Average Precision based on part (APp)               @[mIoU=0.40      ] = {:.3f}'.format(all_ap_p[3]))
        print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(all_ap_p[4]))
        print(' Average Precision based on part (APp)               @[mIoU=0.60      ] = {:.3f}'.format(all_ap_p[5]))
        print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(all_ap_p[6]))
        print(' Average Precision based on part (APp)               @[mIoU=0.80      ] = {:.3f}'.format(all_ap_p[7]))
        print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(all_ap_p[8]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.10      ] = {:.3f}'.format(all_pcp[0]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.20      ] = {:.3f}'.format(all_pcp[1]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.30      ] = {:.3f}'.format(all_pcp[2]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.40      ] = {:.3f}'.format(all_pcp[3]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(all_pcp[4]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.60      ] = {:.3f}'.format(all_pcp[5]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.70      ] = {:.3f}'.format(all_pcp[6]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.80      ] = {:.3f}'.format(all_pcp[7]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.90      ] = {:.3f}'.format(all_pcp[8]))

        eval_results = parsing_result
        eval_results['AP0.5'] = all_ap_p[4]
        eval_results['APvol'] = ap_p_vol
        eval_results['PCP0.5'] = all_pcp[4]

        return eval_results
