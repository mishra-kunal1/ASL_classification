from torchvision.models import resnet50
import torch.nn as nn
import torch  

def pretrained_model(num_classes):
    resnet_model = resnet50(weights=True)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(in_features, num_classes)
    #resnet_model.to('cuda')
    return resnet_model
