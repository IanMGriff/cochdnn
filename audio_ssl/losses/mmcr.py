import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple

import sys


class MMCR(nn.Module):
    def __init__(self, lmbda: float, distributed: bool = True):
        """
        Module for computing the MMCR Loss Function

        Args:
        lmbda: float, weight for the local nuclear norm term
        distributed: bool, whether to network outputs are distributed (i.e. training using DDP)
        """
        super(MMCR, self).__init__()
        self.lmbda = lmbda
        self.distributed = distributed

    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        """
        Computes the MMCR Loss Function

        Args:
        z: Tensor of shape (batch_size, n_aug, emb_dim)

        Returns:
        loss: Tensor of shape (1,)
        loss_dict: dict containing the loss and its components
        """
        z = F.normalize(z, dim=-1)

        # gather across devices into list if more than 1 device
        if self.distributed:
            z_list = [
                torch.zeros_like(z) for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_list, z, async_op=False)
            z_list[torch.distributed.get_rank()] = z

            # append all
            z_local = torch.cat(z_list)

        else:
            z_local = z

        centroids = torch.mean(z_local, dim=1)
        if self.lmbda != 0.0:
            local_nuc = torch.linalg.svdvals(z_local).sum()
        else:
            local_nuc = torch.tensor(0.0)
        global_nuc = torch.linalg.svdvals(centroids).sum()

        batch_size = z_local.shape[0]
        loss = self.lmbda * local_nuc / batch_size - global_nuc

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc.item(),
            "global_nuc": global_nuc.item(),
        }

        return loss, loss_dict
