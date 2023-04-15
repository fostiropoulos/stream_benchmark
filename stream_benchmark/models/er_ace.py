# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from stream_benchmark.utils.buffer import Buffer
from stream_benchmark.models.__base_model import BaseModel


class ErACE(BaseModel):
    name = "er_ace"
    description = "Continual learning via Experience Replay."
    link = "https://arxiv.org/abs/1811.11682"

    def __init__(self, backbone, loss, lr, buffer_size, minibatch_size, **_):
        super(ErACE, self).__init__(backbone, loss, lr)
        self.buffer = Buffer(buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        # self.num_classes = get_dataset(args).N_CLASSES
        self.task = 0
        self.minibatch_size = minibatch_size

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.optimizer.zero_grad()
        # if self.seen_so_far.max() < (self.num_classes - 1):
        mask[:, self.seen_so_far.max() :] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.0)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.minibatch_size, transform=None
            )
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.optimizer.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
