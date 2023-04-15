# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from stream_benchmark.utils.buffer import Buffer
from stream_benchmark.models.__base_model import BaseModel


class Er(BaseModel):
    name = "er"
    description = "Continual learning via Experience Replay."
    link = "https://arxiv.org/abs/1811.11682"

    def __init__(self, backbone, loss, lr, buffer_size, minibatch_size, **_):
        super(Er, self).__init__(backbone, loss, lr)
        self.buffer = Buffer(buffer_size, self.device)
        self.minibatch_size = minibatch_size

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        pass

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.optimizer.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.minibatch_size, transform=None
            )
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])
        return loss.item()
