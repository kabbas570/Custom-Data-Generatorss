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

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        
        self.expand_layer = nn.Linear(embed_dim,16)


        self.transformer1 = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)])
        self.transformer2 = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)])
        self.transformer3 = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)])

        self.transformer1_ = nn.Sequential(*[AttentionBlock(16, 128, num_heads, dropout=dropout)])
        self.transformer2_ = nn.Sequential(*[AttentionBlock(16, 128, num_heads, dropout=dropout)])
        self.transformer3_ = nn.Sequential(*[AttentionBlock(16, 128, num_heads, dropout=dropout)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))
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
        x = self.dropout(x)
        x = x.transpose(0, 1)

        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)

        # Perform classification prediction
        e1=self.expand_layer(x)

        e1 = self.transformer1_(e1)
        e1 = self.transformer2_(e1)
        e1 = self.transformer3_(e1)
        out=patch_image(e1)

        return self.activation(out)
    
def model() -> VisionTransformer:
     model = VisionTransformer(embed_dim=256, hidden_dim=512, num_channels=1, num_heads=8, num_layers=6, num_classes=10, patch_size=4, num_patches=144, dropout=0.2)
     return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
print(model)
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1,48,48) )


