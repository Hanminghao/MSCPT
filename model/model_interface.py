import inspect
import torch
import os
import numpy as np
import pickle
import importlib
import json
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from scipy.special import expit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, gc, seed, num_shots, max_epochs, model_name, n_vpro, wandb_name, txt_result_path, heatmap_path, dataset_name, **kargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.gc = gc
        self.train_loss = 0.
        self.train_acc = 0.
        self.val_loss = 0.
        self.val_acc= 0.
        self.seed = seed
        self.num_shots = num_shots
        self.max_epochs = max_epochs
        self.best_val_score = 0.
        self.model_name = model_name
        self.n_vpro = n_vpro
        self.wandb_name = wandb_name
        self.txt_result_path = txt_result_path
        self.heatmap_path = heatmap_path
        self.dataset_name = dataset_name
    def forward(self, img, train):
        return self.model(img, train)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        data, labels, slide_id = batch
        if self.model_name in ['mscpt']:
            logits, logits_i, logits_t = self(data, True)
            loss1 = self.loss_function(logits, labels)
            loss2 = self.loss_function(logits_i, labels)
            loss3 = self.loss_function(logits_t, labels)
            loss = loss1 + loss2 + loss3
        loss = loss / self.gc
        self.train_loss += loss.item()
        self.manual_backward(loss)

        out_digit = logits.argmax(axis=1)
        correct_num = sum(labels == out_digit).cpu().item()
        self.train_acc += correct_num

        # accumulate gradients of gc batches
        if (batch_idx + 1) % self.gc == 0:
            opt.step()
            opt.zero_grad()
            self.train_loss = 0.
            self.train_acc = 0.
        return {'correct_num': correct_num, 'loss': loss, 'total_num': len(out_digit), 'logits':logits, 'labels':labels}
    
    def training_epoch_end(self, outputs):
        # Make the Progress Bar leave there
        crooret = sum(i['correct_num'] for i in outputs)
        total = sum(i['total_num'] for i in outputs)
        loss = sum([i['loss'] for i in outputs]) / len(outputs)
        logits = torch.cat([i['logits'] for i in outputs], dim=0)
        labels = torch.cat([i['labels'] for i in outputs], dim=0)
        probabilities = torch.softmax(logits.cpu(), dim=1).detach().numpy()
        if logits.shape[-1] != 2:
            auc = roc_auc_score(labels.cpu().numpy(), probabilities, multi_class='ovr')
        else:
            auc = roc_auc_score(labels.cpu().numpy(), expit(logits.detach().cpu().numpy()[:,1]))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train_acc', crooret / total, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train_auc', auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        precision = precision_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro')
        f1 = f1_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro')
        self.log('train_precision', precision, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train_recall', recall, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        

    def validation_step(self, batch, batch_idx):
        data, labels, slide_id = batch
        if self.model_name in ['mscpt']:
            logits, sim_small = self(data, False)
        loss = self.loss_function(logits, labels)
        self.val_loss += loss.item()

        out_digit = logits.argmax(axis=1)
        correct_num = sum(labels == out_digit).cpu().item()
        self.val_acc += correct_num

        if (batch_idx + 1) % self.gc == 0:
            self.val_loss = 0.
            self.val_acc = 0.


        return {'correct_num': correct_num, 'loss': loss, 'total_num': len(out_digit), 'logits':logits, 'labels':labels, 'slide_id':slide_id, 'sim_small':sim_small}


    def validation_epoch_end(self, outputs):
        # Make the Progress Bar leave there
        crooret = sum(i['correct_num'] for i in outputs)
        total = sum(i['total_num'] for i in outputs)
        loss = sum([i['loss'] for i in outputs]) / len(outputs)
        logits = torch.cat([i['logits'] for i in outputs], dim=0)
        labels = torch.cat([i['labels'] for i in outputs], dim=0)
        probabilities = torch.softmax(logits.cpu(), dim=1).detach().numpy()
        if logits.shape[-1] != 2:
            auc = roc_auc_score(labels.cpu().numpy(), probabilities, multi_class='ovr')
        else:
            auc = roc_auc_score(labels.cpu().numpy(), probabilities[:,1])
        acc = crooret / total
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        precision = precision_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro')
        f1 = f1_score(labels.cpu().numpy(), logits.cpu().argmax(axis=1).numpy(), average='macro')
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        if self.dataset_name == 'RCC':
            epoch_score = (f1+acc)/2
        else:
            epoch_score = (f1+auc)/2
        if epoch_score > self.best_val_score:
            self.best_val_score = epoch_score
            self.best_val_auc = auc
            self.best_val_acc = acc
            self.best_val_f1 = f1
            self.best_val_pre = precision
            self.best_val_recall = recall
            save_result = {
                # 'epoch':epoch,
                'acc': acc,
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            with open(os.path.join(self.txt_result_path, self.wandb_name+'.json'), 'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=4)
        self.log('val_best_score', self.best_val_score, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_best_acc', self.best_val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val_best_auc', self.best_val_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val_best_f1', self.best_val_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)


    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
