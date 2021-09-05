"""
Created on Fri Nov 2 2018

@author: cmy
"""

import numpy as np
# crop the image, and feedback a 4-D tensor,
# restrict the input is 2-D tensor

def crop_patch(image, crop_size, overlap):
    im_channel = np.shape(image)[1]
    im_width = np.shape(image)[2]
    im_high = np.shape(image)[3]
    i = int(np.ceil(im_width/(crop_size[0]-overlap[0])))
    j = int(np.ceil(im_high/(crop_size[1]-overlap[1])))
    im_patch = np.zeros((i*j, im_channel,crop_size[0], crop_size[1]), dtype='float32')
    iter = 0
    for im_i in range(0, im_width, crop_size[0]-overlap[0]):
        if im_i + crop_size[0] > im_width:
            temp_i = im_width - crop_size[0]
        else:
            temp_i = im_i
        for im_j in range(0, im_high, crop_size[1]-overlap[1]):
            if im_j + crop_size[1] > im_high:
                temp_j = im_high - crop_size[1]
            else:
                temp_j = im_j
            im_patch[iter, :, :, :] = image[0,:,temp_i:temp_i + crop_size[0], temp_j:temp_j + crop_size[1]]
            iter = iter + 1
    return im_patch


# im_patch is 4-D tensor
# attention: the code just test for overlap=[0,0], others will wrong
# @jit(nopython=True)
def mean_patch(im_patch, restore_size, overlap):
    np_channel = np.shape(im_patch)[1]
    patch_w = np.shape(im_patch)[2]
    patch_h = np.shape(im_patch)[3]
    w_p = patch_w - overlap[0]
    w_h = patch_h - overlap[1]

    count_i = int(np.ceil(restore_size[0] / (patch_w - overlap[0])))
    count_j = int(np.ceil(restore_size[1] / (patch_h - overlap[1])))
    restore = np.zeros((np_channel,restore_size[0], restore_size[1]), dtype='float32')
    count_map = np.zeros((np_channel,restore_size[0], restore_size[1]), dtype='float32')
    map = np.ones((np_channel,patch_w, patch_h), dtype='float32')
    channel = 0
    for i in range(count_i):
        for j in range(count_j):
            if (i*w_p+patch_w > restore_size[0]) and (j*w_h+patch_h > restore_size[1]):
                restore[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] = \
                    restore[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] \
                    + im_patch[channel, :, :, :]
                count_map[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] \
                    = count_map[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] + map

            elif i*w_p+patch_w > restore_size[0] and (j*w_h+patch_h <= restore_size[1]):
                restore[:,(restore_size[0] - patch_w):restore_size[0], (j*w_h):(j*w_h+patch_h)] = \
                    restore[:,(restore_size[0] - patch_w):restore_size[0], (j*w_h):(j*w_h+patch_h)]\
                    + im_patch[channel, :, :, :]
                count_map[:,(restore_size[0] - patch_w):restore_size[0], (j*w_h):(j*w_h+patch_h)] = \
                    count_map[:,(restore_size[0] - patch_w):restore_size[0], (j*w_h):(j*w_h+patch_h)] + map

            elif j*w_h+patch_h > restore_size[1] and (i*w_p+patch_w <= restore_size[0]):
                restore[:,(i*w_p):(i*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]] = \
                    restore[:,(i*w_p):(i*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]]\
                    + im_patch[channel, :, :, :]
                count_map[:,(i*w_p):(i*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]]\
                    = count_map[:,(i*w_p):(i*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]] + map
            else:
                restore[:,(i*w_p):(i*w_p+patch_w), (j*w_h):(j*w_h+patch_h)] = \
                    restore[:,(i*w_p):(i*w_p+patch_w), (j*w_h):(j*w_h+patch_h)] + im_patch[channel, :, :, :]
                count_map[:,(i*w_p):(i*w_p+patch_w), (j*w_h):(j*w_h+patch_h)] = \
                    count_map[:,(i*w_p):(i*w_p+patch_w), (j*w_h):(j*w_h+patch_h)] + map
            channel = channel + 1

    recon = restore / count_map
    return recon

def mean_patch_ycf(im_patch, restore_size, overlap):
    np_channel = np.shape(im_patch)[1]
    patch_w = np.shape(im_patch)[2]
    patch_h = np.shape(im_patch)[3]
    w_p = patch_w - overlap[0]
    w_h = patch_h - overlap[1]

    count_i = int(np.ceil(restore_size[0] / (patch_w - overlap[0])))
    count_j = int(np.ceil(restore_size[1] / (patch_h - overlap[1])))
    restore = np.zeros((np_channel,restore_size[0], restore_size[1]), dtype='float32')
    count_map = np.zeros((np_channel,restore_size[0], restore_size[1]), dtype='float32')
    map = np.ones((np_channel,patch_w, patch_h), dtype='float32')
    channel = 0
    # (count_i - i-1)
    # (count_j - j-1)
    for i in range(count_i):
        for j in range(count_j):
            if ((count_i - i-1)*w_p+patch_w > restore_size[0]) and ((count_j - j-1)*w_h+patch_h > restore_size[1]):
                restore[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] = \
                    restore[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] \
                    + im_patch[(count_i*count_j-1-channel), :, :, :]
                count_map[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] \
                    = count_map[:,(restore_size[0] - patch_w):restore_size[0], (restore_size[1] - patch_h):restore_size[1]] + map

            elif (count_i - i-1)*w_p+patch_w > restore_size[0] and ((count_j - j-1)*w_h+patch_h <= restore_size[1]):
                restore[:,(restore_size[0] - patch_w):restore_size[0], ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] = \
                    restore[:,(restore_size[0] - patch_w):restore_size[0], ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)]\
                    + im_patch[(count_i*count_j-1-channel), :, :, :]
                count_map[:,(restore_size[0] - patch_w):restore_size[0], ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] = \
                    count_map[:,(restore_size[0] - patch_w):restore_size[0], ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] + map

            elif (count_j - j-1)*w_h+patch_h > restore_size[1] and ((count_i - i-1)*w_p+patch_w <= restore_size[0]):
                restore[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]] = \
                    restore[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]]\
                    + im_patch[(count_i*count_j-1-channel), :, :, :]
                count_map[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]]\
                    = count_map[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), (restore_size[1]-patch_h):restore_size[1]] + map
            else:
                restore[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] = \
                    restore[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] + im_patch[(count_i*count_j-1-channel), :, :, :]
                count_map[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] = \
                    count_map[:,((count_i - i-1)*w_p):((count_i - i-1)*w_p+patch_w), ((count_j - j-1)*w_h):((count_j - j-1)*w_h+patch_h)] + map
            channel = channel + 1

    recon = restore / count_map
    return recon

