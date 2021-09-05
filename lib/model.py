from lib.network import *
from lib.utils import *
from lib.STN_model import *

class UDGNet(nn.Module):
    def __init__(self,inchannel):
        super(UDGNet,self).__init__()
        G0 = 32
        ksize = 3
        self.in_channels = inchannel
        self.rain_streak = nn.Sequential(
                        nn.Conv2d(self.in_channels,G0,ksize,padding=(ksize-1)//2,stride=1),
                        ResNet(G0,32),
                        conv_ycf(G0,G0),
                        nn.Conv2d(G0,self.in_channels,ksize,padding=(ksize-1)//2,stride=1)
        )
        self.Image_Net = nn.Sequential(
                        nn.Conv2d(self.in_channels,G0,ksize,padding=(ksize-1)//2,stride=1),
                        ResNet(G0,32),
                        conv_ycf(G0,G0),
                        nn.Conv2d(G0,self.in_channels,ksize,padding=(ksize-1)//2,stride=1)
        )

    def forward(self,input):
        x = input
        rain_streak_residual = self.rain_streak(x)
        rain_streak = input - rain_streak_residual
        Image_reidual = self.Image_Net(x)
        Image = input - Image_reidual

        return {'out_clean':Image,'rain_streak':rain_streak}



