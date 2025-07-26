import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))

class SmoothL1Loss(BaseLoss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.smooth_l1_loss(pred, target, reduction='mean')

class LogDepthLoss(BaseLoss):
    def __init__(self):
        super(LogDepthLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(torch.log(torch.abs(pred - target) + 1))

class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, pred, target):
        return F.mse_loss(pred, target)

class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)
    
class LPIPSLoss(BaseLoss):
    def __init__(self, net='vgg'):
        super(LPIPSLoss, self).__init__()
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, pred, target):
        return self.lpips(pred, target)



class L1_LPIPS_pointshape_Loss(BaseLoss):
    def __init__(self, net='vgg', device='cuda'):
        super(L1_LPIPS_pointshape_Loss, self).__init__()
        self.lpips = lpips.LPIPS(net=net,spatial=False, verbose=False)
        self.L1 = L1Loss()


    def forward(self, pred, target, pointfeats):
        pointfeat_1, pointfeat_2, pointfeat_3 = pointfeats[0], pointfeats[1], pointfeats[2]
        pointshape_diff_loss = torch.mean(torch.norm(pointfeat_1 - pointfeat_2, p=2, dim=1)) + \
                   torch.mean(torch.norm(pointfeat_1 - pointfeat_3, p=2, dim=1)) + \
                   torch.mean(torch.norm(pointfeat_2 - pointfeat_3, p=2, dim=1))

        return 0.4 * torch.abs(self.lpips(pred, target).mean()) + 0.5 * self.L1(pred[target!=0], target[target!=0]) + 0.1 * pointshape_diff_loss 

class L1_pointshape_Loss(BaseLoss):
    def __init__(self):
        super(L1_pointshape_Loss, self).__init__()
        self.L1 = L1Loss()

    def forward(self, pred, target, pointfeats):
        pointfeat_1, pointfeat_2, pointfeat_3 = pointfeats[0], pointfeats[1], pointfeats[2]
        pointshape_diff_loss = torch.mean(torch.norm(pointfeat_1 - pointfeat_2, p=2, dim=1)) + \
                   torch.mean(torch.norm(pointfeat_1 - pointfeat_3, p=2, dim=1)) + \
                   torch.mean(torch.norm(pointfeat_2 - pointfeat_3, p=2, dim=1))

        return 0.9 * self.L1(pred[target!=0], target[target!=0]) + 0.1 * pointshape_diff_loss 

