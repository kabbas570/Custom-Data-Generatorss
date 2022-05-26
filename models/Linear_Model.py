import torch
import torch.nn as nn

class MulticlassClassification(nn.Module):
    def __init__(self):
        super(MulticlassClassification, self).__init__()

        self.layer_1_cl = nn.Linear(224, 1500)
        self.layer_2_cl = nn.Linear(1500, 1000)
        self.layer_3_cl = nn.Linear(1000, 500)
        
        self.layer_1_gen = nn.Linear(224, 1500)
        self.layer_2_gen = nn.Linear(1500, 1000)
        self.layer_3_gen = nn.Linear(1000, 500)
        
        self.layer_out = nn.Linear(1000,3)

    def forward(self, x_cl,x_gen):
        
        ### Clinical data feature extraction ###
        x_cl = self.layer_1_cl(x_cl)    
        x_cl = self.layer_2_cl(x_cl)  
        x_cl = self.layer_3_cl(x_cl)
        
        
        ### Genes data feature extraction ###
        
        x_gen = self.layer_1_gen(x_gen)    
        x_gen = self.layer_2_gen(x_gen)  
        x_gen = self.layer_3_gen(x_gen)
        
        print(x_cl.shape)
        print(x_gen.shape)
        
        
        out = torch.cat((x_cl, x_gen), dim=-1)
        print(out.shape)
        out = self.layer_out(out)
        
        print(out.shape)


       
        return out
    
def model() -> MulticlassClassification:
    model = MulticlassClassification()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
print(model)
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1, 224), (1,224)])
