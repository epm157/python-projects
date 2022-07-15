import torch as t
import torch.nn.functional as F

class ContrastiveLoss(t.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = t.mean((1 - label) * t.pow(euclidean_distance, 2) +
                                  label * t.pow(t.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
