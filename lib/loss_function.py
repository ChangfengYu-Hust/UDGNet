import torch,os
import torch.nn as nn
from torch.autograd import Variable
# from lib.utils import *
from lib.network import *
import torch.optim as optim
import torch.nn.functional as F
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0]

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

class UTV_Gradient_x(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(UTV_Gradient_x, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]

        # _,s,_ = torch.svd(x)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        count_w = 1
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        h_tv = 0
        # gradient_x = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sqrt().sum()
        # L1_loss = torch.where(w_tv >= 1, w_tv - 0.5, loss_part2)
        # loss = s.sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        # return self.TVLoss_weight * loss / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def gradient_y(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    gradient = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
    return gradient
def gradient_x(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    gradient = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
    return gradient


# class Gradient_y(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(Gradient_y, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#
#         # _,s,_ = torch.svd(x)
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
#         count_w = self._tensor_size(x[:, :, :, 1:])
#         count_h = 1
#         # gradient_y =
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sqrt()
#         # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         w_tv = 0
#         return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#
#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]

class TV_Loss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]

        # _,s,_ = torch.svd(x)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return self.TVLoss_weight * 2 * (h_tv / 1 + w_tv / 1)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
# def L1_loss()

class svd_loss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(svd_loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        _,s,_ = torch.svd(x)
        loss = s.sum()
        return self.TVLoss_weight * loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]



def main():
    # x = Variable(torch.FloatTensor([[[1,2],[2,3]],[[1,2],[2,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[3,1],[4,3]],[[3,1],[4,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[1,1,1], [2,2,2],[3,3,3]],[[1,1,1], [2,2,2],[3,3,3]]]).view(1, 2, 3, 3), requires_grad=True)
    x = Variable(
        torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),
        requires_grad=True)
    lr = 0.001
    # addition = TVLoss()
    net = ResBlock()
    net.apply(weights_init_kaiming)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = init_model(model)
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    # criterion2 = UTV_Gradient_x()
    criterion1.cuda()
    criterion2.cuda()
    optimizer = optim.Adam(model.parameters(), lr = lr, eps=1e-4, amsgrad=True)
    for epoch in range(5000):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        input = x.cuda()
        out = net(input)
        h_x = out.size()[2]
        w_x = out.size()[3]
        data_loss = criterion1(out,input)
        prior_loss = criterion1(out[:, :, :, 1:],out[:, :, :, :w_x - 1])
        # prior_loss = criterion2(out)
        loss = 0.1*data_loss + 1*prior_loss
        # gradient = out[:, :, :, 1:] - out[:, :, :, :w_x - 1]
        # loss = Tv_x_loss(out[:, :, :, 1:],out[:, :, :, :w_x - 1])
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(out)
    # torch.save(model.state_dict(), os.path.join(output_folder, 'backup.pth'))

    # print (x)
    # print (z.data)
    # z.backward()
    # print (x.grad)
    # print (x)


if __name__ == "__main__":
    main()