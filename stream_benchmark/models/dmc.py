# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn import functional as F
from stream_benchmark.models.__base_model import BaseModel
from stream_benchmark.datasets.aux_cifar100 import AuxCIFAR100


class DMC(BaseModel):
    name = 'dmc'
    description='Continual learning via Deep Model Consolidation.'
    link = "https://arxiv.org/abs/1903.07864"

    def __init__(self, backbone, loss, lr, dmc, task_start_idx, batch_size, dataset_path,**_):
        super(DMC, self).__init__(backbone, loss, lr)
        self.expert_net = None
        self.old_net = None
        self.task_start_idx = task_start_idx
        self.task = 0
        self.consolidation_lr = dmc['cons_lr']
        self.consolidation_epochs = dmc['cons_epochs']

        self.aux_dataset = AuxCIFAR100(batch_size=batch_size, dataset_path = dataset_path)
        self.aux_train_loader, self.aux_val_loader = self.aux_dataset.get_data_loaders()

    def begin_task(self, *_):
        # Re-initialize net as expert model
        # reset net in begin_task for proper evaluation
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                m.reset_parameters()

    def observe(self, inputs, labels, not_aug_inputs):

        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def end_task(self, *_):
        # Copy net to expert model
        self.expert_net = deepcopy(self.net)
        self.expert_net.eval()
        for param in self.expert_net.parameters():
            param.requires_grad = False

        if self.task > 1:
            # Re-initialize net as student model
            for m in self.net.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    m.reset_parameters()

            optimizer = SGD(self.net.parameters(), lr=self.consolidation_lr)#, momentum=0.5, weight_decay=5e-4)
            scheduler = ReduceLROnPlateau(optimizer, "min", threshold=1e-2, patience=5, verbose=False)

            self.old_net.eval()
            self.expert_net.eval()
            for epoch in range(self.consolidation_epochs):
                e_loss = []
                for i, data in enumerate(self.aux_train_loader):
                    optimizer.zero_grad()
                    x, _ = data
                    x = x.to(self.device)
                    student_logits = self.net(x)
                    with torch.no_grad():
                        old_logits = self.old_net(x)
                        expert_logits = self.expert_net(x)

                        start_idx = self.task_start_idx[self.task]
                        end_idx = self.task_start_idx[self.task+1] if self.task+1 < len(self.task_start_idx) else expert_logits.shape[1]

                        target_logits = torch.cat([old_logits[:, :start_idx],
                                                expert_logits[:, start_idx:end_idx]], dim=1)
                        target_logits -= target_logits.mean(0)
                    # double distillation loss
                    loss = F.mse_loss(student_logits[:, :end_idx], target_logits.detach(), reduction='mean')
                    loss.backward()
                    optimizer.step()
                    e_loss.append(loss.item())

                # evaluate for scheduler
                # val_loss = self.evaluate_aux()
                # print(epoch, val_loss)
                # scheduler.step(val_loss)

        # consolidation finished, copy net to old model
        self.old_net = deepcopy(self.net)
        self.old_net.eval()
        for param in self.old_net.parameters():
            param.requires_grad = False

        self.task += 1

    @torch.no_grad()
    def evaluate_aux(self):
        losses = []
        self.net.eval()
        for data in self.aux_val_loader:
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.net(x)
            loss = self.loss(outputs, y).cpu().numpy()
            losses.append(loss)
        self.net.train()
        return np.array(losses).mean()
