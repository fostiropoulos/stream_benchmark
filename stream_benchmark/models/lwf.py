# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stream_benchmark.datasets.aux_cifar100 import AuxDataset
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from stream_benchmark.models.__base_model import BaseModel


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(BaseModel):
    name = "lwf"
    description = "Continual learning via Learning without Forgetting."
    link = "https://arxiv.org/abs/1606.09282"

    def __init__(self, backbone, loss, lr, n_epochs, batch_size, task_start_idx, lwf, **_):
        super(Lwf, self).__init__(backbone, loss, lr)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.task_start_idx = task_start_idx
        self.current_task = 0
        nc = self.task_start_idx[-1]
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = lwf['alpha']
        self.softmax_temp = lwf['softmax_temp']

    def begin_task(self, train_loader, task_start_idx, *_):
        dl = None
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.classifier.parameters(), lr=self.lr)
            for epoch in range(self.n_epochs):
                for i, data in enumerate(train_loader):
                    inputs, labels = data
                    labels = task_start_idx + labels
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net(inputs, returnt="features")
                    mask = (
                        self.eye[self.task_start_idx[self.current_task + 1] - 1]
                        ^ self.eye[self.task_start_idx[self.current_task] - 1]
                    )
                    outputs = self.net.classifier(feats)[:, mask]
                    loss = self.loss(
                        outputs, labels - self.task_start_idx[self.current_task]
                    )
                    loss.backward()
                    opt.step()

            x, y, logits = [], [], []
            with torch.no_grad():
                for data in train_loader:
                    inputs, labels = data
                    out = self.net(inputs.to(self.device)).cpu()
                    x.append(inputs)
                    y.append(labels)
                    logits.append(out)
            x, y, logits = torch.concat(x), torch.concat(y), torch.concat(logits)
            dl = DataLoader(
                AuxDataset(x, y, logits=logits),
                batch_size=self.batch_size,
                shuffle=True,
            )
        self.net.train()

        self.current_task += 1
        return dl

    def end_task(self, *_):
        pass

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)

        mask = self.eye[self.task_start_idx[self.current_task] - 1]
        loss = self.loss(outputs[:, mask], labels)
        if logits is not None:
            mask = self.eye[self.task_start_idx[self.current_task - 1] - 1]
            loss += self.alpha * modified_kl_div(
                smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                smooth(self.soft(outputs[:, mask]), 2, 1),
            )

        loss.backward()
        self.optimizer.step()

        return loss.item()
