import random
import cv2
import numpy as np

from torchvision.transforms import functional as F

class Compose(object): #对多个数据增强/处理操作进行组合，并按顺序执行这些操作
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints, area, bbox):
        for t in self.transforms:
            image, mask, joints, area, bbox = t(image, mask, joints, area, bbox) #数据增强/处理操作
        return image, mask, joints, area, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ToTensor(object): #将 PIL 图像转换为张量
    def __call__(self, image, mask, joints, area, bbox):
        return F.to_tensor(image), mask, joints, area, bbox #将 PIL 图像转换为张量，同时对张量进行标准化，即将像素值除以 255，将所有像素值缩放到 0 到 1 的范围内

class Normalize(object): #数据预处理，将图像数据进行标准化处理
    def __init__(self, mean, std): #输入图像数据的均值和标准差
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints, area, bbox):
        image = F.normalize(image, mean=self.mean, std=self.std) #图像数据进行标准化处理
        return image, mask, joints, area, bbox

class RandomHorizontalFlip(object): #数据增强，用于随机水平翻转
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index #需要翻转的关键点坐标索引
        self.prob = prob #翻转概率
        self.output_size = output_size 
        self.bbox_flip_index = [1, 0, 3, 2] #边界框坐标索引的翻转顺序

    def __call__(self, image, mask, joints, area, bbox): #数据增强时需要进行的具体操作
        if random.random() < self.prob: #随机生成的数小于概率prob时
            image = image[:, ::-1] - np.zeros_like(image) #将输入的图像image和掩码mask进行水平翻转操作
            mask = mask[:, ::-1] - np.zeros_like(mask)
            joints = joints[:, self.flip_index] #将关键点坐标的x轴坐标进行水平翻转，并将翻转后的x轴坐标值映射回原始坐标系
            joints[:, :, 0] = self.output_size - joints[:, :, 0] - 1
            bbox = bbox[:, self.bbox_flip_index] #将边界框的左右坐标进行交换，并将翻转后的左右坐标值映射回原始坐标系
            bbox[:, :, 0] = self.output_size - bbox[:, :, 0] - 1
        return image, mask, joints, area, bbox

class RandomAffineTransform(object): #随机仿射变换的数据增强方法
    def __init__(self,input_size, output_size, max_rotation,
                 min_scale, max_scale, scale_type, max_translate): #最大旋转角度、最小缩放比例、最大缩放比例、缩放类型和最大平移距离
        self.input_size = input_size
        self.output_size = output_size
        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate

    def _get_affine_matrix(self, center, scale, res, rot=0): #scale：图像缩放比例，一个标量。res：输出图像的大小，一个形如 (height, width) 的元组。rot：旋转角度，一个标量。
        # Generate transformation matrix生成转换矩阵
        h = 200 * scale #将图像按比例缩放到一个固定的高度
        t = np.zeros((3, 3)) #创建一个形如 (3, 3) 的零矩阵 t
        t[0, 0] = float(res[1]) / h #表示将输出图像的宽度缩放为原始图像的宽度与高度比例的乘积，即 res[1] / h
        t[1, 1] = float(res[0]) / h #表示将输出图像的高度缩放为原始图像的宽度与高度比例的乘积，即 res[0] / h。
        t[0, 2] = res[1] * (-float(center[0]) / h + .5) #表示将输出图像水平平移 center[0] * res[1] / h 个像素。
        t[1, 2] = res[0] * (-float(center[1]) / h + .5) #表示将输出图像垂直平移 center[1] * res[0] / h 个像素。
        t[2, 2] = 1 #表示单位矩阵的最后一个元素为 1
        scale = t[0,0]*t[1,1]
        if not rot == 0: #将输入参数 rot 转换为弧度，并计算旋转矩阵 rot_mat。如果 rot 不等于 0，则进行旋转
            rot = -rot  # To match direction of rotation from cropping匹配裁剪的旋转方向
            rot_mat = np.zeros((3, 3)) #创建一个形如 (3, 3) 的零矩阵 rot_mat
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad) #计算旋转角度的正弦值和余弦值（sn 和 cs）
            rot_mat[0, :2] = [cs, -sn] #将旋转角度的正弦值和余弦值分别赋值给 rot_mat 的第一行和第二行。
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1 #将 rot_mat 的最后一个元素设为 1。
            # Need to rotate around center 计算绕图像中心旋转的仿射变换矩阵
            t_mat = np.eye(3) #平移矩阵，3x3的单位矩阵
            t_mat[0, 2] = -res[1]/2 #第一行第三个元素
            t_mat[1, 2] = -res[0]/2 #第二行第三个元素，第三行是[0,0,1]
            t_inv = t_mat.copy() #平移矩阵是t_mat的一个镜像，将图像中心平移到原点，这样旋转就可以直接绕原点进行
            t_inv[:2, 2] *= -1 #第一行和第二行的第三个元素*-1，第三行是[0,0,1]
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t))) #由t_inv、rot_mat和t_mat三个矩阵相乘得到
        return t, scale #t：一个形如 (3, 3) 的 NumPy 数组，表示仿射变换矩阵

    def _affine_joints(self, joints, mat): #对关节点进行仿射变换，joints是关节点坐标列表，mat是仿射变换矩阵
        joints = np.array(joints) #将输入的 joints 数组转换为 numpy 数组
        shape = joints.shape
        joints = joints.reshape(-1, 2) #将joints重新整形为 2 列的数组
        return np.dot(np.concatenate( #将joints和一个全为1，列数为1的矩阵按列拼接之后，通过np.dot函数计算仿射变换矩阵和列向量点积。然后将变形后的数组重新整形为输入形状
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints, area, bbox): #数据增强器
        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long': #决定了输入图像的比例是基于长尺寸还是短尺寸
            scale = max(height, width)/200
            print("###################please modify range")
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale #max_scale和min_scale参数用于在一定范围内随机缩放图像。
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation #随机旋转图像

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale) #随机地水平和垂直平移
            center[0] += dx
            center[1] += dy

        mat_output, _ = self._get_affine_matrix(
            center, scale, (self.output_size, self.output_size), aug_rot
        )
        mat_output = mat_output[:2]
        mask = cv2.warpAffine((mask*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) / 255 #根据计算出的变换矩阵变换输入图像和mask
        mask = (mask > 0.5).astype(np.float32) #二值化处理

        joints[:, :, 0:2] = self._affine_joints(
            joints[:, :, 0:2], mat_output
        )
        bbox = self._affine_joints(bbox, mat_output)

        mat_input, final_scale = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )
        mat_input = mat_input[:2]
        area = area*final_scale
        image = cv2.warpAffine(image, mat_input, (self.input_size, self.input_size))

        return image, mask, joints, area, bbox