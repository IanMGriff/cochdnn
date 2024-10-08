
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import lightning as L

import architectures
from metrics import calculate_accuracy
from loss_functions import MMCR_Loss
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.jsinV3DataLoader_precombined import jsinV3_precombined_paired
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS

class ModelWithFrontEnd(nn.Module):
    def __init__(self,front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(self, x):
        x, _ = self.front_end(x, None)
        feature, out, logits = self.model(x)
        return feature, out, logits    

class LitAudioSSL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        # Init audio transforms 
        self.transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=config['audio_transforms']['low_snr'],
                                          high_snr=config['audio_transforms']['high_snr']),
                at.DBSPLNormalizeForegroundAndBackground(dbspl=config['audio_transforms']['dbspl']),
                at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
            ])

        # Get audio config and init representation 
        self.audio_config = AUDIO_INPUT_REPRESENTATIONS[config['audio_rep']['name']]
        self.audio_rep = at.AudioToAudioRepresentation(**self.audio_config)

        # Get audio model from config kwargs
        self.model = architectures.__dict__[self.config['model']['arch_name']](**self.config['model']['arch_kwargs'])
        
        if config['audio_rep']['on_gpu']:
            # If computing rep on gpu, compose rep and model in same forward pass for convenience
            self.model = ModelWithFrontEnd(self.audio_rep, self.model)
        else:
            # if computing rep on cpu, add rep as last stage of audio transforms
            self.transforms = at.AudioCompose([
                self.transforms,
                self.audio_rep
            ])
        
        # init losses 
        # if torch.distributed.is_initialized():
        #     distributed=True

        self.ssl_task = self.config['hparas']['ssl_task']
        self.mmcr_loss = MMCR_Loss(distributed=True) # comeback to see if distrubuted needs to be true here 
        self.lambda_mmcr = self.config['hparas']['lambda_mmcr']
        self.opt_supervised_task = self.config['model']['arch_kwargs']['supervised']
        if self.opt_supervised_task:
            self.class_loss = nn.CrossEntropyLoss()

    def _step(self, batch, batch_idx, step_type):
        spec_11, spec_12, spec_21, spec_22, labels_1, labels_2 = batch

        # pass pairs through model 
        _, out_11, logits_11 = self.model(spec_11)
        _, out_12, logits_12 = self.model(spec_12)
        _, out_21, logits_21 = self.model(spec_21)
        _, out_22, logits_22 = self.model(spec_22)

        # concat pairs with same equivariances and get mmcr los
        outs_1 = torch.cat([out_11, out_21], dim=0)
        outs_2 = torch.cat([out_12, out_22], dim=0)
        
        loss_mmcr = self.mmcr_loss(outs_1, outs_2)
        self.log(f"{step_type}_mmcr_loss", loss_mmcr.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        class_loss = 0 
        if self.opt_supervised_task:
        # get classification loss
            class_loss_11 = self.class_loss(logits_11, labels_1)
            class_loss_12 = self.class_loss(logits_12, labels_1)
            class_loss_21 = self.class_loss(logits_21, labels_2)
            class_loss_22 = self.class_loss(logits_22, labels_2)
            class_loss = class_loss_11 + class_loss_12 + class_loss_21 + class_loss_22
            class_loss = class_loss / 4.0
            self.log(f"{step_type}_class_loss", class_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        total_loss = self.lambda_mmcr * loss_mmcr + class_loss
        self.log(f"{step_type}_total_loss", total_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # calc acc 
        acc = 0 
        acc += calculate_accuracy(logits_11, labels_1).item()
        acc += calculate_accuracy(logits_12, labels_1).item()
        acc += calculate_accuracy(logits_21, labels_2).item()
        acc += calculate_accuracy(logits_22, labels_2).item()
        acc /= 4  
        self.log(f"{step_type}_class_acc", acc, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)


        # add acc to log 
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self.train_dataloader.dataset._rotate_splits()
        print(f"Updated rotation: {self.train_dataloader.dataset.rotate_index}")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        # Optimizer
        opt = getattr(torch.optim, self.config['hparas']['optimizer'])
        self.optimizer = opt(self.model.parameters(), lr=self.config['hparas']['lr'])      
        return [self.optimizer]

    def forward(self, x):
        """
        PL required forward wrapper. Enables calling model in two ways:
        1) standard call in .py scripts
            model = LitAudioSSL(args)
            outs = model(inputs)
        2) inside this lightning module's methods as self (eg in _step)
            outs = self(inputs) # self is self.forward, and is same as self.model.forward 
        """
        return self.model(x)

    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_paired(root=self.config['data']['root'], train=True, transform=self.transforms)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
        )
        return self.train_dataloader
    
    def val_dataloader(self):
        dataset = jsinV3_precombined_paired(root=self.config['data']['root'], train=False, transform=self.transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )
        return dataloader

