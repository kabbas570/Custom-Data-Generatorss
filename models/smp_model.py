import segmentation_models_pytorch as smp
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 1
ACTIVATION = 'sigmoid' 

########  create segmentation model with pretrained encoder   ######

DeepLabV3Plus_Model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=NUM_CLASSES, 
    activation=ACTIVATION,
)

from torchsummary import summary

DeepLabV3Plus_Model.to(device=DEVICE,dtype=torch.float)
summary(DeepLabV3Plus_Model,(3,512,512))


# Configure data preprocessing  #####



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


print(preprocessing_fn)
