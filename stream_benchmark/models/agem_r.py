# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from stream_benchmark.utils.buffer import Buffer
from stream_benchmark.models.gem import overwrite_grad
from stream_benchmark.models.gem import store_grad
from stream_benchmark.models.agem import project
from stream_benchmark.models.__base_model import BaseModel

class AGemr(BaseModel):
    name = 'agem_r'
    description='Continual learning via A-GEM, leveraging a reservoir buffer.'
    link = 'https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html'

    def __init__(self, backbone, loss, lr, buffer_size, minibatch_size, **_):
        super(AGemr, self).__init__(backbone, loss, lr)

        self.buffer = Buffer(buffer_size, self.device)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.current_task = 0
        self.minibatch_size = minibatch_size

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        pass

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.minibatch_size)
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.optimizer.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
