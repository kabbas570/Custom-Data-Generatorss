import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Model_LA(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Model_LA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // 2)
        
        self.up0 = Up(2048, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.activation = torch.nn.Sigmoid()
        
        self.up_ = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1_):
        ##encoder 1 ##
        x1 = self.inc(x1_)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        

        return self.activation(x6) 
    
# def model() -> Model_LA:
#     model = Model_LA()
#     return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(1, 640,640)])



from pthflops import count_ops

# Create a network and a corresponding input
device = 'cuda:0'
model = Model_LA()
model.to(device=DEVICE,dtype=torch.float)
inp = torch.rand(1,1,80,80).to(DEVICE)

# Count the number of FLOPs
count_ops(model, inp)
