# from __future__ import print_function, division
from torch import nn
import torch


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return bce_loss + (1 - dice_coef)


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        return bce_loss


class weightedBCELoss(nn.Module):
    def __init__(self):
        super(weightedBCELoss, self).__init__()

    def forward(self, input, target):
        pred = input
        truth = target

        # weighted BCE loss
        weight_1 = torch.Tensor([0.8]).double().cuda()
        weight_0 = torch.Tensor([0.2]).double().cuda()

        label = truth.double()
        output = torch.clamp(pred, min=1e-10, max=1-1e-10).double()
        loss = weight_1 * (label * torch.log(output)) + weight_0 * ((1 - label) * torch.log(1 - output))
        bce_loss = torch.neg(torch.mean(loss))

        return bce_loss


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return dice_coef


class BatchBCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BatchBCEDiceLoss, self).__init__()

    def forward(self, input, target):
        total_score = 0
        for i in range(input.size()[0]):
            total_score += BCEDiceLoss()(input[i, :, :, :], target[i, :, :, :])

        return total_score
