# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from stream_benchmark.utils.buffer import Buffer
from stream_benchmark.models.gem import overwrite_grad
from stream_benchmark.models.gem import store_grad
from stream_benchmark.models.__base_model import BaseModel

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(BaseModel):
    name = 'agem'
    description='Continual learning via A-GEM.'
    link='https://arxiv.org/abs/1812.00420'

    def __init__(self, backbone, loss, lr, buffer_size, minibatch_size, task_start_idx, **_):
        super(AGem, self).__init__(backbone, loss, lr)

        self.buffer = Buffer(buffer_size, self.device)
        self.grad_dims = []
        self.examples_per_task = buffer_size // (len(task_start_idx) - 1)
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.minibatch_size = minibatch_size
        self.buffer_size = minibatch_size

    def begin_task(self, *_):
        pass

    def end_task(self, train_loader, task_start_idx, *_):
        x, y = [], []
        for data in train_loader:
            inputs, labels = data
            x.append(inputs)
            y.append(labels + task_start_idx)
        x, y = torch.concat(x), torch.concat(y)
        # sub-sample
        perm = torch.randperm(x.shape[0])
        x = x[perm][:self.examples_per_task]
        y = y[perm][:self.examples_per_task]

        self.buffer.add_data(
            examples=x,
            labels=y,
        )

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

        return loss.item()
