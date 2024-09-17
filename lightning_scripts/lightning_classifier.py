
import torch
from torch import nn
from torchmetrics.classification import Accuracy 
import torch.nn.functional as F
import lightning as L

from metrics import calculate_accuracy
import robustness.audio_models as architectures
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS
from robustness.audio_functions.jsinV3DataLoader_precombined import jsinV3_precombined_all_signals

class ModelWithFrontEnd(nn.Module):
    def __init__(self,front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(self, x):
        x, _ = self.front_end(x, None)
        return self.model(x)

class LitWordAudioSetModel(L.LightningModule):
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
        self.model = architectures.__dict__[self.config['model']['arch_name']](**self.config['model']['arch_params'])

        if config['audio_rep']['on_gpu']:
            # If computing rep on gpu, compose rep and model in same forward pass for convenience
            self.model = ModelWithFrontEnd(self.audio_rep, self.model)
        else:
            # if computing rep on cpu, add rep as last stage of audio transforms
            self.transforms = at.AudioCompose([
                self.transforms,
                self.audio_rep
            ])
    
        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=config['hparas']['batch_size'])
        # get accuracy metrics per task - requires module dict for torchmetrics 
        self.train_accuracy = torch.nn.ModuleDict({task_key: Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.val_accuracy = torch.nn.ModuleDict({task_key: Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.accuracy = {'train': self.train_accuracy, 'val': self.val_accuracy}

    def _step(self, batch, batch_idx, step_type):
        audio, label_dict = batch

        # logits will be dict - keys for each task
        logits = self.model(audio)
      
        # get classification loss
        loss = self.multi_task_loss(logits, label_dict)
        # add losses to log
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
  
        # calc acc 
        for task, task_logits in logits.items():
            task_acc = self.accuracy[step_type][task](task_logits, label_dict[task])
            self.log(f"{step_type}_{task}_acc", task_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        # Optimizer
        opt = getattr(torch.optim, self.config['hparas']['optimizer'])
        self.optimizer = opt(self.model.parameters(), lr=self.config['hparas']['lr'])     
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['hparas']['step_lr']) 
        return [self.optimizer], [self.schedule]

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
        dataset = jsinV3_precombined_all_signals(root=self.config['data']['root'], train=True, transform=self.transforms)
        dataset.target_keys = self.config['data']['target_keys']
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
        dataset = jsinV3_precombined_all_signals(root=self.config['data']['root'], train=False, transform=self.transforms)
        dataset.target_keys = self.config['data']['target_keys']
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )
        return dataloader

