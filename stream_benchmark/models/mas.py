# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from stream_benchmark.models.__base_model import BaseModel


class MAS(BaseModel):
    name = 'mas'
    description = 'Continual learning via MAS.'
    link = "https://arxiv.org/abs/1711.09601"

    def __init__(self, backbone, loss, lr, batch_size, mas, **_):
        super(MAS, self).__init__(backbone, loss, lr)

        self.older_params = None
        self.importance = self.init_importance()
        self.e_lambda = mas['e_lambda']
        self.gamma = mas['gamma']
        self.batch_size = batch_size

    def penalty(self):
        if self.older_params is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = 0
            for n, p in self.net.named_parameters():
                if n in self.importance.keys():
                    penalty += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
            return penalty

    def init_importance(self):
        return {n: torch.zeros(p.shape).to(self.device) for n, p in self.net.named_parameters()
                if p.requires_grad and 'classifier' not in n}

    def begin_task(self, *_):
        pass

    def end_task(self, train_loader, task_start_idx, *_):
        # get old params
        self.older_params = {n: p.clone().detach() for n, p in self.net.named_parameters()
                             if p.requires_grad and 'classifier' not in n}

        # get param importance
        importance = self.init_importance()
        n_samples_batches = 0
        self.net.train()
        for batch in train_loader:
            self.net.zero_grad()
            inputs, labels = batch
            labels = task_start_idx + labels
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output = self.net(inputs)
            loss = torch.norm(output, p=2, dim=1).mean()
            loss.backward()
            n_samples_batches += 1

            for n, p in self.net.named_parameters():
                if n in self.importance and p.grad is not None:
                    importance[n] += p.grad.abs() * len(labels)

        n_samples = n_samples_batches * self.batch_size
        importance = {n: (p / n_samples) for n, p in self.importance.items()}

        # merge fisher information
        for n in self.importance.keys():
            # As in original code: add prev and new
            self.importance[n] = self.gamma * self.importance[n] + (1 - self.gamma) * importance[n]

    def observe(self, inputs, labels, not_aug_inputs):

        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.optimizer.step()

        return loss.item()
