import torch
import torch.nn as nn
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

def patch_image(x):
    
    _,B,_=x.shape
    x=x.view([B,144,1,4,4])
    x=x.view([B,12,12,1,4,4])
    x = x.permute(0, 3, 1, 4, 2, 5) # [B, H', W', C, p_H, p_W]
    x = x.reshape(B,1,48,48)
    return x
    
class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
class VisionTransformer(nn.Module):

    def __init__(self, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches):

        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), 64)
        
        self.expand_layer1 = nn.Linear(64,128)
        self.expand_layer2 = nn.Linear(128,256)
        
        self.expand_layer = nn.Linear(256,128)
        self.expand_layer_ = nn.Linear(128,64)
        self.expand_layer_1 = nn.Linear(64,16)


        self.transformer1 = nn.Sequential(*[AttentionBlock(64, 128, num_heads)])
        self.transformer2 = nn.Sequential(*[AttentionBlock(128, 256, num_heads)])
        self.transformer3 = nn.Sequential(*[AttentionBlock(256, 512, num_heads)])

        self.transformer1_ = nn.Sequential(*[AttentionBlock(256, 128, num_heads)])
        self.transformer2_ = nn.Sequential(*[AttentionBlock(128, 64, num_heads)])
        self.transformer3_ = nn.Sequential(*[AttentionBlock(64, 16, num_heads)])
        self.transformer4_ = nn.Sequential(*[AttentionBlock(16, 16, num_heads)])

    

        # Parameters/Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,64))
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        
        x = self.input_layer(x)

        # Add and positional encoding
        y=self.pos_embedding[:,:T]
        x = x + y

        # Apply Transforrmer
        print(x.shape)
        x = x.transpose(0, 1)
        print(x.shape)

        x = self.transformer1(x)
        print(x.shape)
        x = self.expand_layer1(x)
        x = self.transformer2(x)
        print(x.shape)
        x = self.expand_layer2(x)
        x = self.transformer3(x)
        
        print(x.shape)

        #e1=self.expand_layer(x)
        #print(e1.shape)

        e1 = self.transformer1_(x)
        print(e1.shape)
        e1=self.expand_layer(e1)
        e1 = self.transformer2_(e1)
        print(e1.shape)
        e1=self.expand_layer_(e1)
        # e1 = self.transformer2_(e1)
        e1 = self.transformer3_(e1)
        print(e1.shape)
        e1=self.expand_layer_1(e1)
        e1 = self.transformer4_(e1)
        print(e1.shape)
        # print(e1.shape)
        out=patch_image(e1)

        return  self.activation(out)
    
def model() -> VisionTransformer:
     model = VisionTransformer (num_channels=1, num_heads=8, num_layers=6, num_classes=10, patch_size=4, num_patches=144)
     return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1,48,48) )

import numpy as np
a=np.load('/Users/kabbas570gmail.com/Documents/Challenge/del_/data/gt/train_11_19_11.npy')
a=np.expand_dims(a, axis=0)
a=np.expand_dims(a, axis=0)
a=torch.tensor(a)
x = img_to_patch(a, patch_size=4)
x=x.permute(1,0,2)
out=patch_image(x)

i=a.numpy()[0,0,:,:]
q=out.numpy()[0,0,:,:]
