import logging
import os
import os.path
from random import choice
import torch
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train, transform=None, target_generator=None):
        super(PoseDataset, self).__init__()
        self.root = cfg.DATASET.ROOT #数据集所在目录 
        self.dataset = cfg.DATASET.DATASET
        if self.dataset == 'crowdpose':
            from crowdposetools.coco import COCO
        else:
            from pycocotools.coco import COCO
        self.image_set = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
        self.is_train = is_train #当前是否为训练集
        self.transform = transform  #数据增强的变换
        self.coco = COCO(os.path.join(self.root, 'annotations', '{}_{}.json'.format(self.dataset, self.image_set))) #coco类
        self.ids = list(self.coco.imgs.keys())   #所有图像的id列表

        #过滤掉出错的或者不好的
        if is_train:
            if self.dataset == 'coco':
                self.filter_for_annotations()
            else:
                self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0] 
            self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS   #关键点个数
            self.output_size = cfg.DATASET.OUTPUT_SIZE #网络输出尺寸
            self.heatmap_generator = target_generator   #标签生成器

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        if self.dataset == 'coco': images_dir = os.path.join(images_dir, '{}2017'.format(self.image_set))
        return os.path.join(images_dir, file_name)

    def filter_for_annotations(self, min_kp_anns=1):
        print('filter for annotations (min kp=%d) ...', min_kp_anns)

        def filter_image(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            anns = [ann for ann in anns if not ann.get('iscrowd')]
            if not anns:
                return False
            kp_anns = [ann for ann in anns
                       if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])]
            return len(kp_anns) >= min_kp_anns

        self.ids = [image_id for image_id in self.ids if filter_image(image_id)]

    def __getitem__(self, index): #和enumerat枚举相关的函数
        coco = self.coco  #获取coco数据集对象
        img_id = self.ids[index]  #获取指定下标 index 所对应的图片的 ID
        file_name = coco.loadImgs(img_id)[0]['file_name'] #从 coco 中获取该图片的文件名 file_name
        img = cv2.imread(                    #使用 OpenCV 的 imread 函数加载该图片的像素数据，并存储在变量 img 中
            self._get_image_path(file_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将其颜色空间由 BGR 转换为 RGB

        results = {}
        if self.is_train:
            ann_ids = coco.getAnnIds(imgIds=img_id) #获得了该图像的所有标注信息的id。
            target = coco.loadAnns(ann_ids) #载入了该图像的所有标注信息。
            anno = [obj for obj in target] #将标注信息存储在列表anno中
            img_info = self.coco.loadImgs(img_id)[0] #载入了图像的信息
            # mask
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
        else:
            results['image'] = torch.from_numpy(img)
            results['image_id'] = img_id

        return results

    def __len__(self):
        return len(self.ids)

    def get_inst_annos(self, centers, keypoints, area, bbox):
        ind_vis = []
        area_idx = np.argsort(area.squeeze()) #遍历物体面积从小到大排序后的物体，对每个物体进行实例分割注释提取
        inst_coords, inst_heatmaps, inst_masks = [], [], []
        for i in area_idx:
            inst_coord = []
            center = centers[i, 0] #获取第i个物体的中心点坐标
            if center[2] < 1: continue
            x, y = int(center[0]), int(center[1])
            if x < 0 or x >= self.output_size or y < 0 or y >= self.output_size: continue
            # rand center point in 3x3 grid
            new_x = x + choice([-1, 0, 1]) 
            new_y = y + choice([-1, 0, 1])                    
            if new_x < 0 or new_x >= self.output_size or new_y < 0 or new_y >= self.output_size:       
                new_x = x                        
                new_y = y                    
            x, y = new_x, new_y
            
            if [y, x] in ind_vis: continue
            inst_coord.append([y, x])
            ind_vis.append([y, x])
            inst_coords.append(np.array(inst_coord)) # [[1], [1,2], [1,2,3]...]
            inst_heatmap, inst_mask = self.heatmap_generator(keypoints[i:i+1, :, :], bbox[i:i+1, :, :])
            inst_heatmaps.append(inst_heatmap[None, :, :, :])
            inst_masks.append(inst_mask[None, :, :, :])
        return inst_coords, inst_heatmaps, inst_masks

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str