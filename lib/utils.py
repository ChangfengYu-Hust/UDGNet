import math,random
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import cv2
from lib.AffineGridGen import *
import torch.nn.functional as  F


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def transform_no_crop(input_tensor,angle_nor):
    angle = 90 - ((angle_nor) * 90 + 45)
    alpha = math.radians(angle[0])
    (h, w) = input_tensor.shape[2:]
    nW = math.ceil(h * math.fabs(math.sin(alpha)) + w * math.cos(alpha))
    nH = math.ceil(h * math.cos(alpha) + w * math.fabs(math.sin(alpha)))
    x1 = torch.zeros(size=(input_tensor.shape[0],input_tensor.shape[1],nH,nW))
    for i in range(input_tensor.size()[0]):
        alpha = math.radians(angle[i])
        (h,w) = input_tensor.shape[2:]
        theta = torch.tensor([
            [math.sin(-alpha), math.cos(alpha), 0],
            [math.cos(alpha), math.sin(alpha), 0]
        ], dtype=torch.float).cuda()
        theta = theta.unsqueeze(0)
        nW = math.ceil(h * math.fabs(math.sin(alpha)) + w * math.cos(alpha))
        nH = math.ceil(h * math.cos(alpha) + w * math.fabs(math.sin(alpha)))
        img = input_tensor[i, :, :, :]
        img = img.unsqueeze(0)
        g = AffineGridGen(nH, nW, aux_loss=True)
        grid_out, aux = g(theta)
        grid_out[:, :, :, 0] = grid_out[:, :, :, 0] * nW / w
        grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW / h
        out = F.grid_sample(img, grid_out)
        x1[i, :, :, :] = out.squeeze(0)
    return x1
def transform_crop(input_tensor,angle_nor):
    angle = 90 - ((angle_nor) * 90 + 45)
    x1 = torch.clone(input_tensor)
    for i in range(input_tensor.size()[0]):
        alpha = math.radians(angle[i])
        theta = torch.tensor([
            [math.cos(alpha), math.sin(-alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0]
        ], dtype=torch.float).cuda()
        img = input_tensor[i,:,:,:]
        img = img.unsqueeze(0)
        N, C, H, W = img.size()
        grid = F.affine_grid(theta.unsqueeze(0), torch.Size((N, C, W, H)))
        img = F.grid_sample(img, grid)
        x1[i,:,:,:] = img.squeeze(0)
    return x1






def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def img_processing(img):
  # 灰度化
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = img
  if len(gray.shape)== 3:
      print("gray error")
  # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
  x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
  y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
  absX = cv2.convertScaleAbs(x)  # 转回uint8
  return absX
def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out





class param():
    def __init__(self, image):
        if image.shape[2] > 1:
            im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y = im_yuv[:, :, 0]
        else:
            y = image
        y = y/255
        image = y
        self.SNR = float(np.var(image)*(0.2*np.random.random_sample(1)+0.3))
        g = np.round(0.5+np.round(8*np.random.random_sample(1)))
        if np.mod(g, 2) == 0:
            g = g+1
        g2 = np.round(0.5 + np.round(1.5 * np.random.random_sample(1)))
        if np.mod(g2, 2) == 0:
            g2 = g2 + 1
        self.g_size = (g, g)
        self.g_size2 = (g2, g2)
        self.sv = 0.1+0.3*np.random.random_sample(1)
        self.sv2 = 0.1 + 0.3 * np.random.random_sample(1)
        self.angle = 45+90*np.random.random_sample(1)
        self.length = 30+20*np.random.random_sample(1)

if __name__ == "__main__":
    data_path = ""
    image = cv2.imread(data_path)
    image = image[np.newaxis,:,:,:]
    angle = 0