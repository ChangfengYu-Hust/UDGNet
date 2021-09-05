# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797


import torch
import torch.nn as nn
import numpy as np
from lib.dataset import *
import torch.nn.init as init
import functools
import torch.nn.functional as F
from torch.autograd import Variable

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class ResBlock(nn.Module):
    def __init__(self, Channels, kSize=3):
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
class conv_ycf(nn.Module):
    def __init__(self, in_Channels, out_channels):
        super(conv_ycf, self).__init__()
        self.in_channels = in_Channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )
        # self.conv_ycf1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.conv_ycf2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1)


    def forward(self, x):
        x = self.conv(x)
        return x
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=12, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       (ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
class ResNet(nn.Module):
    def __init__(self, growRate0, nConvLayers, kSize=3):
        super(ResNet, self).__init__()
        G0 = growRate0
        self.C = nConvLayers
        self.convs = []
        convs = []
        for i in range (self.C):
            # self.res = ResBlock(G0)
            convs.append(ResBlock(G0))
        self.convs = nn.Sequential(*convs)



    def forward(self,x):
        # feat_output = []
        for i in range(self.C):
            x = self.convs[i].forward(x)
            # feat_output.append(x)

        return x
class down(nn.Module):
    def __init__(self, in_ch, out_ch,stride):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(stride),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class angle_estimation(nn.Module):
    def __init__(self,in_channels):
        super(angle_estimation, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.code = double_conv(in_channels,32)

        self.down1 = down(32,64,2)
        self.down2 = down(64,128,4)
        self.down3 = down(128,64,4)
        self.down4 = down(64,32,4)
        self.angle_layer = nn.Conv2d(32,1,kernel_size=3,padding=1)
    def forward(self,input):
        x = input
        x1 = self.conv1x1(x)    #(n,c,128,128)
        x1 = self.code(x1)      #(n,c,128,128)
        x2 = self.down1(x1)     #(n,c,64,64)
        x3 = self.down2(x2)     #(n,c,16,16)
        x4 = self.down3(x3)     #(n,c,4,4)
        x5 = self.down4(x4)     #(n,c,1,1)
        out_put = self.angle_layer(x5)
        return out_put

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class discriminator_model(nn.Module):
    def __init__(self,in_channels):
        super(discriminator_model, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.code = double_conv(in_channels,64)

        self.down1 = down(64,128)  #(b,128,64,64)
        self.down2 = down(128,256)  #(b,128,32,32)
        self.down3 = down(256,128)  #(b,128,16,16)
        self.down4 = down(128, 64)  # (b,64,8,8)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, padding=1),  # (b, 32, 128, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.AvgPool2d(4, stride=4,padding=0),  # (b, 32, 32, 32)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, padding=1),  # （b,64,32, 32）
        #     nn.LeakyReLU(0.2),
        #     nn.AvgPool2d(4, stride=2,padding=1)  # （b,64,8, 8）
        # )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = input
        x1 = self.conv1x1(x)
        x1 = self.code(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = x5.view(x5.size(0), -1)
        x = self.fc(x)
        return x

class generator_Dncnn(nn.Module):
    def __init__(self, inchannels, outchannels, num_of_layers=17):
        super(generator_Dncnn, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=inchannels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = x - self.dncnn(x)
        return out

class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list

class rains_streak_Net(nn.Module):
    def __init__(self):
        super(rains_streak_Net,self).__init__()
        G0 = 32
        ksize = 3
        self.in_channels = 32
        self.out_channels = 32
        self.rain_mask = nn.Sequential(
                        nn.Conv2d(1,G0,ksize,padding=(ksize-1)//2,stride=1),
                        ResNet(G0,32),
                        conv_ycf(G0,G0),
                        nn.Conv2d(G0,1,ksize,padding=(ksize-1)//2,stride=1)
        )
    def forward(self, input):
        rain_mask = self.rain_mask(input)
        clean = torch.sub(input,rain_mask)
        return clean

class Image_net(nn.Module):
    def __init__(self):
        super(Image_net,self).__init__()
        G0 = 64
        ksize = 3
        self.in_channels = 32
        self.out_channels = 32
        self.Image = nn.Sequential(
            nn.Conv2d(1, G0, ksize, padding=(ksize - 1) // 2, stride=1),
            ResNet(G0, 32),
            conv_ycf(G0, G0),
            nn.Conv2d(G0, 1, ksize, padding=(ksize - 1) // 2, stride=1)
        )

    def forward(self, input):
        Clean_Image = self.Image(input)
        return Clean_Image


def init_model(model, init_type='normal'):

    def weights_init_normal(m, std=0.02):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                # print('initializing [%s] ...' % classname)
                init.normal_(m.weight.data, 0.0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(m, scale=1):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                # print('initializing [%s] ...' % classname)
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.constant_(m.weight.data, 1.0)
            m.weight.data *= scale
            init.constant_(m.bias.data, 0.0)

    def weights_init_orthogonal(m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                # print('initializing [%s] ...' % classname)
                init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def init_weights(net, init_type='kaiming', scale=1, std=0.02):
        # scale for 'kaiming', std for 'normal'.
        print('initialization method [%s]' % init_type)
        if init_type == 'normal':
            weights_init_normal_ = functools.partial(weights_init_normal, std=std)
            net.apply(weights_init_normal_)
            return net
        elif init_type == 'kaiming':
            weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
            net.apply(weights_init_kaiming_)
            return net
        elif init_type == 'orthogonal':
            net.apply(weights_init_orthogonal)
            return net
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    return init_weights(model, init_type)
