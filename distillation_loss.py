import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TEMPERATURE


class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cls_logits, dist_logits, teacher_logits, labels):
        loss_cls = self.ce(cls_logits, labels)  # cls_logits: students prediction for each class. 
                                                 # we use CE for cls as CE expects one hot taregt encoding ie class 0 = [1,0]   
                                                #crossentropy = losscls= -log(ptrue_class)

        loss_dist = F.kl_div(
            F.log_softmax(dist_logits / TEMPERATURE, dim=1), # Teacher outputs raw scores or logits
            F.softmax(teacher_logits / TEMPERATURE, dim=1),# After adding softwmax and tempurature it gives probability
            reduction="batchmean"#                          KL divergence measures how far apart these two distribution are
        ) * (TEMPERATURE ** 2)

        return loss_cls + loss_dist


