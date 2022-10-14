import torchvision.models as models
import torch
import torch.nn as nn

class Detection_Model_MobileNet(nn.Module):
    
    def __init__(self):
        super(Detection_Model_MobileNet, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=False)
        self.backbone=  self.backbone.features
        self.flatten_layer=nn.Flatten()
        
        ## FC Layers for CLassification ######
        self.fc1_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=47040, out_features=1000),
            nn.ReLU())
        
        self.fc2_classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=8),
            )
        
    def forward(self, x):
        x_features = self.backbone(x)
        x_features=self.flatten_layer(x_features)
        
        print(x_features.shape)
        
        classifier_out = self.fc1_classifier(x_features)
        classifier_out = self.fc2_classifier(classifier_out)
        
        return classifier_out  
    

def model() -> Detection_Model_MobileNet:
    model = Detection_Model_MobileNet()
    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3,224,224))
