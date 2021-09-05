from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

class AffineGridGenFunction(Function):
    def __init__(self, height, width,lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        t_height = np.zeros(shape=(1,self.height),dtype=float)
        t_width = np.zeros(shape=(1,self.width),dtype=float)
        temp_height = np.expand_dims(np.arange(-1, 1, 2.0/self.height),0)
        temp_width = np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0)
        t_height[0,:] = temp_height[0,0:self.height]
        t_width[0,:] = temp_width[0,0:self.width]
        self.grid[:,:,0] = np.expand_dims(np.repeat(t_height, repeats = self.width, axis = 0).T, 0)*height/width
        self.grid[:,:,1] = np.expand_dims(np.repeat(t_width, repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                data1 = self.batchgrid.view(-1, self.height*self.width, 3)
                data2 = torch.transpose(input1, 1, 2)
                output1 = torch.bmm(data1,data2)
                output1 = output1.view(-1, self.height, self.width, 2)
                # output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output1

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
            #print('gradout:',grad_output.size())
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        return grad_input1


class AffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1,0,0], [0,1,0]], dtype=np.float32)).cuda()
            batch_identity = torch.zeros([input.size(0), 2,3]).cuda()
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss,1)
            # loss = torch.sum(loss,2)

            return self.f(input), loss.view(-1,1)