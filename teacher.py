import torch
import torch.nn as nn
from torchvision import models, transforms


class ResNet50Teacher(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes) # Replace Final default layer form 2408 to 2 

    def forward(self, x):      #when called teacher(images) Pytorch calls teacher,forward(images). This passes image batch to RES50. Returns logits not probability
        return self.model(x) # ie input:  (B, 3, 224, 224)
                             # output: (B,2) / pain/no paid                               

