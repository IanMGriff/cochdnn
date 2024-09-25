
import torch 
from torch import nn 
import torch.nn.functional as F


class MMCR_Loss(nn.Module):
    def __init__(self, distributed=False):
        super().__init__()
        self.distributed = distributed 
    
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        # gather does nothing if not in distributed training environment
        z1, z2 = self.gather(z1), self.gather(z2)
        c = (z1 + z2) / 2.0

        return -1.0 * torch.linalg.svdvals(c).sum()

    def gather(self, tensor):
        if torch.distributed.is_initialized():
            tensor_list = [torch.zeros_like(tensor) for i in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list, tensor, async_op=False)
            tensor_list[torch.distributed.get_rank()] = tensor
            return torch.cat(tensor_list)
        else:
            return tensor


class Barlow_Loss(nn.Module):
    def __init__(self, lmbda: float, out_dim=8192, scale_loss: float = 0.048, distributed: bool = False):
        super(Barlow_Loss, self).__init__()
        self.lmbda = lmbda 
        self.bn = nn.BatchNorm1d(out_dim, affine=False)
        self.scale_loss = scale_loss
        self.distributed = distributed

    def forward(self, z1, z2):
        # sum the cross-correlation matrix between all gpus
        bs = z1.shape[0]
        if self.distributed:
            bs *= torch.distributed.get_world_size()

        c_inv = self.bn(z1).T @ self.bn(z2)
        c_inv.div_(bs)
        if self.distributed:
            torch.distributed.all_reduce(c_inv)

        on_diag_inv = torch.diagonal(c_inv).add_(-1).pow_(2).sum()
        off_diag_inv = off_diagonal(c_inv).pow_(2).sum()
        loss = on_diag_inv + self.lmbda * off_diag_inv
        loss = loss * self.scale_loss

        return loss



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
