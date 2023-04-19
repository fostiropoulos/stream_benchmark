# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from stream_benchmark.models.__base_model import BaseModel


class SI(BaseModel):
    name = "si"
    description = "Continual Learning Through Synaptic Intelligence."
    link = "https://arxiv.org/abs/1703.04200"

    def __init__(self, backbone, loss, lr, si, **_):
        super(SI, self).__init__(backbone, loss, lr)

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

        self.c = si['c']
        self.xi = si['xi']
        self.lr = lr

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (
                self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)
            ).sum()
            return penalty

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += (self.small_omega / (
            (self.net.get_params().data - self.checkpoint.to(self.device)) ** 2 + self.xi
        ))

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.c * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.optimizer.step()

        self.small_omega += self.lr * self.net.get_grads().data ** 2

        return loss.item()
