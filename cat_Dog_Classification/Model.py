import torch .nn as nn
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def truple_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.Blocck1_Conv =  double_conv(3,64)
        self.Blocck2_Conv =  double_conv(64,128)
        self.Blocck3_Conv =  truple_conv(128,256)
        self.Blocck4_Conv =  truple_conv(256,512)
        self.Blocck5_Conv =  truple_conv(512,512)
        
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)


        self.max_pool_2d = nn.MaxPool2d(2)
        self.activation = torch.nn.Sigmoid()
        self.Drop_Out= nn.Dropout(p=0.30)
        
    def forward(self, x):
        x = self.Blocck1_Conv(x)
        x = self.max_pool_2d(x)
        x = self.Blocck2_Conv(x)
        x = self.max_pool_2d(x)
        x = self.Blocck3_Conv(x)
        x = self.max_pool_2d(x)
        x = self.Blocck4_Conv(x)
        x = self.max_pool_2d(x)
        x = self.Blocck5_Conv(x)
        x = self.max_pool_2d(x)
        x = x.flatten(1)
        
        x = self.fc1(x)
        x = self.Drop_Out(x)
        x = self.fc2(x)
        x = self.Drop_Out(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
def model() -> VGG16:
      model = VGG16 ()
      return model

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3,224,224) )   