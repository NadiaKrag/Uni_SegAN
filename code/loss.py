import torch

import torch.nn as nn
import torch.nn.functional as F

def binary_dice_coef(output, label, eps):
    assert(output.shape == label.shape)
    output = output.view(-1)
    label = label.view(-1)
    return ((2*(output*label).sum()+eps)/(output.sum()+label.sum()+eps))

class MulticlassDiceLoss(nn.Module):
    """
    Author https://github.com/huangmozhilv/u2net_torch
    """
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, args, output, label, num_class):
        label = F.one_hot(label.long(), num_classes=num_class).permute(0,-1,1,2,3)
        if args.cuda:
            label.cuda()
        total_dice = list()
        for i in range(num_class):
            dice = binary_dice_coef(output[:,i,...], label[:,i,...], eps=0.000001)
            total_dice.append(dice)
        mean_dice = torch.div(sum(total_dice), num_class)
        return 1-mean_dice, mean_dice, total_dice