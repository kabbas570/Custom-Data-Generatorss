import torch
import torch.nn as nn

class Dense_Block(nn.Module):
  def __init__(self, in_channels):
    super(Dense_Block, self).__init__()
    self.relu = nn.ReLU(inplace = True)
    self.bn = nn.BatchNorm2d(num_features = in_channels)

    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
  def forward(self, x):
    bn = self.bn(x) 
    conv1 = self.relu(self.conv1(bn))
    conv2 = self.relu(self.conv2(conv1))
    # Concatenate in channel dimension
    c2_dense = self.relu(torch.cat([conv1, conv2], 1))
    conv3 = self.relu(self.conv3(c2_dense))
    c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
   
    conv4 = self.relu(self.conv4(c3_dense)) 
    c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
   
    conv5 = self.relu(self.conv5(c4_dense))
    c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
   
    return c5_dense

class Transition_Layer(nn.Module): 
  def __init__(self, in_channels, out_channels):
    super(Transition_Layer, self).__init__() 
    
    self.relu = nn.ReLU(inplace = True) 
    self.bn = nn.BatchNorm2d(num_features = out_channels) 
    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
    self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) 
  def forward(self, x): 
    bn = self.bn(self.relu(self.conv(x))) 
    out = self.avg_pool(bn) 
    return out 

class Transition_Layer_U(nn.Module): 
  def __init__(self, in_channels, out_channels):
    super(Transition_Layer_U, self).__init__() 
    
    self.relu = nn.ReLU(inplace = True) 
    self.bn = nn.BatchNorm2d(num_features = out_channels) 
    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
    self.up_samp = nn.UpsamplingBilinear2d(scale_factor=2) 
  def forward(self, x): 
    bn = self.bn(self.relu(self.conv(x))) 
    out = self.up_samp(bn) 
    return out 
class UNet(nn.Module): 
  def __init__(self): 
    super(UNet, self).__init__() 
  
    self.lowconv = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, padding = 3, bias = False) 
    self.relu = nn.ReLU()
    
    # Make Dense Blocks 
    self.denseblock1 = self._make_dense_block(Dense_Block, 64)  #64 is input features to this block
    self.denseblock2 = self._make_dense_block(Dense_Block, 64)
    self.denseblock3 = self._make_dense_block(Dense_Block, 128)
    self.denseblock4 = self._make_dense_block(Dense_Block, 256)
    ### up block
    
    self.denseblock1_u = self._make_dense_block(Dense_Block, 160) 
    self.denseblock2_u = self._make_dense_block(Dense_Block, 288)
    self.denseblock3_u = self._make_dense_block(Dense_Block, 224)
    self.denseblock4_u = self._make_dense_block(Dense_Block, 192)
    # Make transition Layers 
    self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
    self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 256) 
    self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
    
    ## upsapling transition
    self.transitionLayer1u = self._make_transition_layer(Transition_Layer_U, in_channels = 160, out_channels = 128) 
    self.transitionLayer2u = self._make_transition_layer(Transition_Layer_U, in_channels = 160, out_channels = 64) 
    self.transitionLayer3u = self._make_transition_layer(Transition_Layer_U, in_channels = 160, out_channels = 32)
    # Classifier 
    self.conv_last = nn.Conv2d(160, 1, 1)
    self.activation = torch.nn.Sigmoid()

 
  def _make_dense_block(self, block, in_channels): 
    layers = [] 
    layers.append(block(in_channels)) 
    return nn.Sequential(*layers) 
  def _make_transition_layer(self, layer, in_channels, out_channels): 
    modules = [] 
    modules.append(layer(in_channels, out_channels)) 
    return nn.Sequential(*modules) 
  def forward(self, x): 
    x = self.relu(self.lowconv(x)) 
    x_d1 = self.denseblock1(x) 
    x_t1 = self.transitionLayer3(x_d1) 
    x_d2 = self.denseblock2(x_t1) 
    x_t2 = self.transitionLayer1(x_d2) 
    x_d3 = self.denseblock3(x_t2) 
    x_t3 = self.transitionLayer2(x_d3) 
    x_d4 = self.denseblock4(x_t3)
    
    
    u_d1=self.denseblock1_u(x_d4)
    u_u1=self.transitionLayer1u(u_d1)
    
    cat_3=torch.cat([u_u1, x_d3], dim=1) 
    
    u_d2=self.denseblock2_u(cat_3)
    u_u2=self.transitionLayer2u(u_d2)
    
    cat_2=torch.cat([u_u2, x_d2], dim=1) 
    
    u_d3=self.denseblock3_u(cat_2)
    u_u3=self.transitionLayer3u(u_d3)
    
    cat_1=torch.cat([u_u3, x_d1], dim=1) 
    u_d4=self.denseblock4_u(cat_1)
    
    out = self.conv_last(u_d4)
    
    return self.activation(out)
