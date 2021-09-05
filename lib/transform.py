import torch,cv2,os
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import data,filters
import matplotlib.pyplot as plt
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from skimage.measure.simple_metrics import compare_psnr
from utils import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def functional_conv2d(im):
    sobel_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect
def functional_conv2d_y(im):
    sobel_kernel = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect
''''
def main():
    # 读入一张图片，并转换为灰度图
    im = Image.open('/home/ubuntu/2TB/YCF/dataset/real_data/real_dataset1/13.png').convert('L')
    # 将图片数据转换为矩阵
    im = np.array(im, dtype='float32')

    dst = img_processing(im)
    # cv2.imshow("test1",dst)
    cv2.waitKey(0)
    thresh = filters.threshold_otsu(dst)  # 返回一个阈值
    thresh = 45
    for i in range (dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] < thresh:
                dst[i][j] = 0
    # dst = (im >= thresh) *255 # 根据阈值进行分割
    dst = dst.astype('uint8')
    cv2.imshow("test2",dst)
    cv2.waitKey(0)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    # 开运算：先腐蚀，后膨胀
    open_img = cv2.dilate(cv2.erode(dst, kernel), kernel)
    open_img = open_img.astype('uint8')
    # 闭运算：先膨胀，后腐蚀
    close_img = cv2.erode(cv2.dilate(dst, kernel), kernel)

    cv2.namedWindow("test",0)
    cv2.imshow("test",open_img)
    cv2.waitKey(0)
    # plt.imshow(dst)
    # plt.show()

    dst = dst.astype('float32')
    open_img = open_img.astype('float32')
    # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
    open_img = torch.from_numpy(open_img.reshape((1, 1, im.shape[0], im.shape[1])))
    # 边缘检测操作
    # edge_detect = nn_conv2d(im)
    edge_detect = functional_conv2d(open_img)
    cv2.imshow("test",edge_detect)
    cv2.waitKey(0)
    # edge_detect = edge_detect.astype('float32')
    # edge_detect = torch.from_numpy(edge_detect.reshape((1, 1, edge_detect.shape[0], edge_detect.shape[1])))
    # edge_detect = functional_conv2d_y(edge_detect)
    # cv2.imshow("test",edge_detect)
    # cv2.waitKey(0)
    # 将array数据转换为image
    im = Image.fromarray(edge_detect)
    # image数据转换为灰度模式
    im = im.convert('L')
    # 保存图片
    im.save('edge.jpg', quality=95)


'''

import torch,math,glob,re
import torch.nn as nn
from torch.autograd import Variable
from AffineGridGen import *
# from lib.utils import *
import torch.optim as optim
from torchvision import transforms
from utils import *

def tryint(s):                       #//将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):               # //将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):    #//以分割后的list为单位进行排序
    return sorted(v_list, key=str2int)

class ResBlock(nn.Module):
    def __init__(self, Channels=2, kSize=3):
        super(ResBlock, self).__init__()
        self.channels = Channels
        self.relu  = nn.ReLU()

        self.res = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels),
        )


    def forward(self, x):
        x = (x+self.res(x))
        return x

class TV_x_Loss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_x_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]

        # _,s,_ = torch.svd(x)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        h_tv = 0
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # loss = s.sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        # return self.TVLoss_weight * loss / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def test_loss():

    x = Variable(
        torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),
        requires_grad=True)
    lr = 0.001
    # addition = TVLoss()
    net = ResBlock()
    net.apply(weights_init_kaiming)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = init_model(model)
    criterion = TV_x_Loss()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr = lr, eps=1e-4, amsgrad=True)
    for epoch in range(5000):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        input = x.cuda()
        out = net(input)
        loss = criterion(out)
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(out)
def param2theta(param, w, h):
    param = np.linalg.inv(param)
    # theta = np.zeros([2,3])
    # theta= param[0:2,:]
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + param[0, 0] + param[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + param[1, 0] + param[1, 1] - 1
    return theta

def transform(input,angle_nor,name):
    # angle = 90 - ((angle_nor) * 90 + 45)
    angle = 45
    alpha = math.radians(angle)
    #
    # # alpha = alpha
    image = input
    (h, w) = image.shape[:2]
    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    img = image
    # LF, img = decomposition(image)
    N, C, h1, w1 = img.size()

    # theta = torch.tensor([
    #     [math.cos(alpha), math.sin(-alpha), 0],
    #     [math.sin(alpha), math.cos(alpha), 0]
    # ], dtype=torch.float)
    theta = torch.tensor([
        [math.sin(-alpha), math.cos(alpha), 0],
        [math.cos(alpha), math.sin(alpha), 0]
    ], dtype=torch.float)
    theta = theta.unsqueeze(0)
    nW = math.ceil(h*math.fabs(math.sin(alpha))+w*math.cos(alpha))
    nH = math.ceil(h*math.cos(alpha)+w*math.fabs(math.sin(alpha)))

    g = AffineGridGen(nH, nW, aux_loss=True)
    grid_out, aux = g(theta)
    grid_out[:,:,:,0] = grid_out[:,:,:,0]*nW / w
    grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW/h
    out = F.grid_sample(img, grid_out)
    # print((out.size()))
    # out.backward(out.data)
    # print(input.grad.size())


    # grid = F.affine_grid(theta.unsqueeze(0), torch.Size((N, C, nW, nH)))
    # grid[:,:,:,0] = grid[:,:,:,0]*nW / w
    # grid[:, :, :, 1] = grid[:, :, :, 1]  * nH/h
    # out = F.grid_sample(img, grid)
    alpha = -alpha
    N, C, h, w = out.size()
    nW = math.ceil(h*math.fabs(math.sin(alpha))+w*math.cos(alpha))
    nH = math.ceil(h*math.cos(alpha)+w*math.fabs(math.sin(alpha)))

    theta = torch.tensor([
        [math.sin(-alpha), math.cos(alpha), 0],
        [math.cos(alpha), math.sin(alpha), 0]
    ], dtype=torch.float)
    theta = theta.unsqueeze(0)


    g = AffineGridGen(nH, nW, aux_loss=True)
    grid_out, aux = g(theta)
    grid_out[:,:,:,0] = grid_out[:,:,:,0]*nW / w
    grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW/h
    HF2 = F.grid_sample(out, grid_out)
    out2 = torch.clone(image)
    out2[:,:,:,:] = HF2[:,:,int((nH-h1)/2):int((nH+h1)/2),int((nW-w1)/2):int((nW+w1)/2)]
    # out2 = out2 + LF
    out1 = out.cpu().squeeze(0).numpy().transpose(1,2,0)*255
    out1 = out1.astype('uint8')
    out2 = out2.cpu().squeeze(0).numpy().transpose(1,2,0)*255
    out2 = out2.astype('uint8')
    # (w2, h2) = out2.shape[:2]
    # final_img = np.zeros_like(input)
    # final_img[:,:,:] = out2[int((nH-h1)/2):int((nH+h1)/2),int((nW-w1)/2):int((nW+w1)/2),:]
    # final_img = final_img + LF
    # residual = image - out2
    # input = input.cpu().squeeze(0).numpy().transpose(1,2,0)*255
    # input = input.astype('uint8')

    # imgs = np.hstack([input,out])
    print("the img:%s, the angle_nor:%f, the angle:%f"%(name,angle_nor,angle))
    cv2.namedWindow("test",0)
    cv2.imshow("test",out1)
    cv2.waitKey(0)
    cv2.imwrite("result/transoform/transform1.png",out1)
    cv2.imshow("test",out2)
    cv2.waitKey(0)
    cv2.imwrite("result/transoform/transform2.png", out2)
    PSNR = compare_psnr(out2,input)
    print(PSNR)
    # cv2.imshow("test",final_img)
    # cv2.waitKey(0)
    # cv2.imwrite("result/transoform/transform3.png", final_img)
    # cv2.imshow("test",residual)
    # cv2.waitKey(0)
    # cv2.imwrite("result/transoform/transform4.png", residual)
def get_residue(img):
    max_channel = torch.max(img, dim=1, keepdim=True)
    min_channel = torch.min(img, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel
def decomposition(x):
    res = get_residue(x)
    eps_list = [0.001, 0.0001]
    radiux = [2, 4, 8, 16]
    gf = GuidedFilter(radiux[2],eps_list[0])
    LF = gf(res,x)
    HF = x - LF
    return LF, HF
if __name__ == "__main__":
    img_path = "result/img"
    angle_path = "result/angle"
    imgs_list = glob.glob(img_path+"/*.png")
    imgs_list = sort_humanly(imgs_list)
    angle_list = glob.glob(angle_path+"/*.npy")
    angle_list = sort_humanly(angle_list)
    for i in range(len(imgs_list)):
        image = cv2.imread(imgs_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        name = imgs_list[i].split('/')[-1]
        angle = np.load(angle_list[i])
        # LF,HF = decomposition(image)
        HF_2 = transform(image,angle,name)
        # Image = LF +HF_2
        # Residual = image - Image
        # cv2.namedWindow("test2",0)
        # cv2.imshow("test2",Residual)
        # cv2.waitKey(0)


