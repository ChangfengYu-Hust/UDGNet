import os
import os.path
import numpy as np
import random
# import h5py
import torch
import cv2,re, imageio
import glob
import torch.utils.data as udata
from lib.utils import data_augmentation
from lib.utils import *
from PIL import Image
from _tkinter import _flatten

def tryint(s):                       #//将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):               # //将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):    #//以分割后的list为单位进行排序
    return sorted(v_list, key=str2int)

class MyData_cityscape(udata.Dataset):
    def __init__(self,input_root,clean_root,angle_root,transforms,patch_size,batch_size,repeat,channel):
        switch = os.listdir(input_root)
        switch = sort_humanly(switch)
        self.input_image = []
        self.angle_data = []
        self.real_image = []
        for i in range(len(switch)):
            input_image = glob.glob(os.path.join(input_root,switch[i])+'/*.png') + glob.glob(os.path.join(input_root,switch[i])+'/*.jpg')
            input_image = sort_humanly(input_image)
            if input_image!=None:
                self.input_image.append(input_image)
            angle_data = glob.glob(os.path.join(angle_root,switch[i])+'/*.npy')
            angle_data = sort_humanly(angle_data)
            if angle_data!=None:
                self.angle_data.append(angle_data)
            clean_image = glob.glob(os.path.join(clean_root,switch[i])+'/*.png') + glob.glob(os.path.join(clean_root,switch[i])+'/*.jpg')
            clean_image = sort_humanly(clean_image)
            if clean_image!=None:
                self.real_image.append(clean_image)
        random.shuffle(self.real_image)
        self.input_image = _flatten(self.input_image)
        self.angle_data = _flatten(self.angle_data)
        self.real_image = _flatten(self.real_image)
        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)

        self.transforms = transforms
        self.patch_size = patch_size
        self.channel = channel

    def __getitem__(self, index):
        input_index = index % len(self.input_image)
        real_index = index % len(self.real_image)
        # lr, hr = self._load_file(index)

        input_image_path = self.input_image[input_index]
        angle_label_path = self.angle_data[input_index]
        real_data_path = self.real_image[real_index]

        input_data = cv2.imread(input_image_path,-1)
        real_data = cv2.imread(real_data_path,-1)
        angle_data = np.load(angle_label_path)
        # angle_data = np.zeros(shape=[1,1,1],dtype=float)
        angle_data = angle_data.reshape([1,1,1])
        file_name = input_image_path.split("/")[-1]

        if self.patch_size !=0:
            row = np.random.randint(input_data.shape[0]-self.patch_size)
            col = np.random.randint(input_data.shape[1]-self.patch_size)
            
            row2 = np.random.randint(real_data.shape[0]-self.patch_size)
            col2 = np.random.randint(real_data.shape[1]-self.patch_size)
            input_data = input_data[row:row+self.patch_size, col:col+self.patch_size, :]
            real_data = real_data[row2:row2 + self.patch_size, col2:col2 + self.patch_size, :]

        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
            real_yuv = cv2.cvtColor(real_data, cv2.COLOR_RGB2YCrCb)
            real_y = real_yuv[:, :, 0]
            
        else:
            input_y = input_data
            real_y = real_data
            im_yuv = input_data

        input_y = Image.fromarray(input_y)
        real_y = Image.fromarray(real_y)
        # im_yuv = Image.fromarray(im_yuv)
        if self.transforms:
            input_y = self.transforms(input_y)
            real_y = self.transforms(real_y)
        return {'rain_data': input_y, 'angle_label':angle_data, 'clean_data':real_y,'filename':file_name,'input_yuv':im_yuv}

    def __len__(self):
        return self.length


