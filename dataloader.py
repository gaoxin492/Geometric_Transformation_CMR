import random
import shutil
import cv2
import torch
from PIL import Image
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np


class MyData(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_path = os.listdir(self.root_dir)
        self.transform = transform
        self.classes2d = ['00', '01', '02', '03', '10', '11', '12', '13']

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        label = img_name.split('@')[-1][0:-4]
        label_tensor = torch.zeros(8)
        label_tensor[self.classes2d.index(label)] = 1.0
        img_item_path = os.path.join(self.root_dir, img_name)
        img_idx = np.array(Image.open(img_item_path), dtype='uint8')
        # 自适应直方图均衡
        img_idx = img_idx.reshape(3,img_idx.shape[0],img_idx.shape[1])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_res = np.zeros_like(img_idx)
        for i in range(img_idx.shape[0]):
            img_res[i,:, :] = clahe.apply(img_idx[i,:, :])
        img_res = Image.fromarray(img_res.reshape(img_res.shape[1],img_res.shape[2],3))
        # 作用transform
        if self.transform is not None:
            img_res = self.transform(img_res)
        return img_res, label_tensor

    def __len__(self):
        return len(self.img_path)


class GenericData:

    def __init__(self, save_path, load_path, split_ratio, dim):
        self.save_path = save_path
        self.load_path = load_path
        self.split_ratio = split_ratio  #[0.8,0.2]
        self.dim = dim
        self.classes2d = ['00', '01', '02', '03', '10', '11', '12', '13']

    def generic_data(self):
        train_save_path = os.path.join(self.save_path, 'train')
        test_save_path = os.path.join(self.save_path, 'test')
        for path in [train_save_path,  test_save_path]:
            if os.path.exists(path):  # 判断文件夹是否存在，如果存在，先清空
                shutil.rmtree(path)
            os.makedirs(path)  # 新增空文件夹

        if self.dim == 2:
            classes = self.classes2d
        else:
            raise ValueError("需要对3d图像进行变换吗？")
        img_path = os.listdir(self.load_path)
        img_path_all = dict()
        for img_name in img_path:
            img_allqueue = nib.load(os.path.join(self.load_path, img_name))
            width, height, queue = img_allqueue.dataobj.shape
            for i in range(queue):
                img = img_allqueue.dataobj[:,:,i]
                for k in range(self.dim * 4):
                    axis_flip = int(classes[k][0])
                    rotation = int(classes[k][1]) * 90
                    img_path_all[img_name + '@{}@{}'.format(i, classes[k])] = Geo_Transform_img(img, axis_flip,rotation)

        img_train, img_test = self.dict_split_shuffle(img_path_all)
        for key in img_train:
            plt.imsave(os.path.join(train_save_path,f'{key}.jpg'), img_train[key], cmap='gray')
        for key in img_test:
            plt.imsave(os.path.join(test_save_path,f'{key}.jpg'), img_test[key], cmap='gray')

    def dict_split_shuffle(self, img_path_all):
        tr_size = int(len(img_path_all) * self.split_ratio[0])
        keys = list(img_path_all.keys())
        random.shuffle(keys)
        img_train = dict([(i,img_path_all[i]) for i in keys[:tr_size]])
        img_test = dict([(i, img_path_all[i]) for i in keys[tr_size:]])

        return img_train, img_test

def show_img(path):
    img = nib.load(path)
    width, height, queue = img.dataobj.shape
    num = 1
    for i in range(queue):
        img_arry = img.dataobj[:, :, i]
        plt.subplot(2, 3, num)
        plt.imshow(img_arry, cmap='gray')
        num += 1
    plt.show()


def rotate_img(img, rot, axes):
    """
    :param img: Array of two or more dimensions.
    :param rot: Degrees of the array is rotated.
    :param axes: The array is rotated in the plane defined by the axes.
    Axes must be different.(0,1),(1,2),(0,2)
    """
    if rot in [0, 90, 180, 270]:
        k = rot / 90
        return np.rot90(img, k, axes)
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270')


def Geo_Transform_img(img, axis_flip, rotation):
    """
    :param img: Array of two or three dimensions.
    :param axis_flip: int, how many aixs should be fipped.
    :param rotation:  rotation degrees in [0,90,180,270]
    :return:
    """
    if axis_flip == 0:  # 没有坐标轴翻转
        return rotate_img(img, rotation, (0, 1))
    elif axis_flip == 1:  # 有一个坐标轴翻转
        img = np.transpose(img)
        return rotate_img(img, rotation, (0, 1))
    # elif axis_flip == 2:  # 有两个坐标轴翻转（说明是3d的情形）
    #     img = np.transpose(img)
    #     return rotate_img(img, rotation, (0, 2))

# Set the paths of the datasets.
MyoPS_C0_dir = 'datasets\MyoPS\C0'
MyoPS_LGE_dir = 'datasets\MyoPS\LGE'
MyoPS_T2_dir = 'datasets\MyoPS\T2'

MyoPS_C0_split_dir = 'datasets\MyoPS\C0_split'
MyoPS_LGE_split_dir = 'datasets\MyoPS\LGE_split'
MyoPS_T2_split_dir = 'datasets\MyoPS\T2_split'

data_generate = GenericData(save_path=MyoPS_T2_split_dir,load_path=MyoPS_T2_dir,split_ratio=[0.8,0.2],dim=2)

data_generate.generic_data()