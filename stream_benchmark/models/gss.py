# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from stream_benchmark.utils.gss_buffer import Buffer as Buffer
from stream_benchmark.models.__base_model import BaseModel


class Gss(BaseModel):
    name = "gss"
    description = "Gradient based sample selection for online continual learning"
    link = "https://arxiv.org/abs/1903.08671"

    def __init__(self, backbone, loss, lr, buffer_size, minibatch_size, gss, **_):
        super(Gss, self).__init__(backbone, loss, lr)
        gss_minibatch_size = gss['gss_minibatch_size']
        self.buffer = Buffer(
            buffer_size,
            self.device,
            gss_minibatch_size
            if gss_minibatch_size is not None
            else minibatch_size,
            self,
        )
        self.alj_nepochs = gss['batch_num']
        self.minibatch_size = minibatch_size

    def get_grads(self, inputs, labels):
        self.net.eval()
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.optimizer.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        pass

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        self.buffer.drop_cache()
        self.buffer.reset_fathom()

        for _ in range(self.alj_nepochs):
            self.optimizer.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.minibatch_size, transform=None
                )
                tinputs = torch.cat((inputs, buf_inputs))
                tlabels = torch.cat((labels, buf_labels))
            else:
                tinputs = inputs
                tlabels = labels

            outputs = self.net(tinputs)
            loss = self.loss(outputs, tlabels)
            loss.backward()
            self.optimizer.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        return loss.item()
