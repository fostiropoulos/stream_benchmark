# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from stream_benchmark.models.__base_model import BaseModel
import torch
import math
from tqdm import tqdm


class JointGCL(BaseModel):
    name = 'joint_gcl'
    description = 'Joint training: a strong, simple baseline.'
    link = "https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html"

    def __init__(self, backbone, loss, lr, n_epochs, batch_size, **_):
        super(JointGCL, self).__init__(backbone, loss, lr)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        # reinit network
        self.net.reset_parameters()
        self.net.to(self.device)
        self.net.train()
        self.opt = SGD(self.net.parameters(), lr=self.lr)

        # gather data
        all_data = torch.cat(self.old_data)
        all_labels = torch.cat(self.old_labels)

        # train
        # progress = tqdm(total=self.n_epochs * math.ceil(len(all_data) / self.batch_size))
        for e in range(self.n_epochs):
            rp = torch.randperm(len(all_data))
            for i in range(math.ceil(len(all_data) / self.batch_size)):
                inputs = all_data[rp][i * self.batch_size:(i+1) * self.batch_size]
                labels = all_labels[rp][i * self.batch_size:(i+1) * self.batch_size]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels.long())
                loss.backward()
                self.optimizer.step()
                # progress.set_description(
                #     f"{e}/{self.n_epochs} loss: {loss.item():.4f}"
                # )
                # progress.update(1)

    def observe(self, inputs, labels, not_aug_inputs):
        self.old_data.append(inputs.data)
        self.old_labels.append(labels.data)
        return 0
