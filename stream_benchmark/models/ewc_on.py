# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from stream_benchmark.models.__base_model import BaseModel


class EwcOn(BaseModel):
    name = 'ewc_on'
    description='Continual learning via online EWC.'
    link = "https://arxiv.org/pdf/1805.06370.pdf"

    def __init__(self, backbone, loss, lr, ewc_on, batch_size, **_):
        super(EwcOn, self).__init__(backbone, loss, lr)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.e_lambda = ewc_on['e_lambda']
        self.gamma = ewc_on['gamma']
        self.batch_size = batch_size

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def begin_task(self, *_):
        pass

    def end_task(self, train_loader, task_start_idx, *_):
        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(train_loader):
            inputs, labels = data
            labels = task_start_idx + labels
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(inputs)
            loss = - F.nll_loss(self.logsoft(output), labels,
                                reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= len(train_loader)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    def observe(self, inputs, labels, not_aug_inputs):

        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.optimizer.step()

        return loss.item()
