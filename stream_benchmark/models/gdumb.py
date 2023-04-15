# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stream_benchmark.models.__base_model import BaseModel
from torch.optim import SGD, lr_scheduler
from stream_benchmark.utils.buffer import Buffer
from stream_benchmark.utils.augmentations import cutmix_data


def fit_buffer(self, config, buffer):
    epochs = config['fitting_epochs']
    max_lr = config['maxlr']
    min_lr = config['minlr']
    cutmix_alpha = config['cutmix_alpha']
    batch_size = config['batch_size']
    for epoch in range(epochs):

        optimizer = SGD(
            self.net.parameters(),
            lr=max_lr,
        )
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=min_lr
        )

        if epoch <= 0:  # Warm start of 1 epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = max_lr * 0.1
        elif epoch == 1:  # Then set to maxlr
            for param_group in optimizer.param_groups:
                param_group["lr"] = max_lr
        else:
            scheduler.step()

        all_inputs, all_labels = buffer.get_data(
            len(buffer.examples), transform=None
        )

        while len(all_inputs):
            optimizer.zero_grad()
            buf_inputs, buf_labels = (
                all_inputs[:batch_size],
                all_labels[:batch_size],
            )
            all_inputs, all_labels = (
                all_inputs[batch_size :],
                all_labels[batch_size :],
            )

            if cutmix_alpha is not None:
                inputs, labels_a, labels_b, lam = cutmix_data(
                    x=buf_inputs.cpu(), y=buf_labels.cpu(), alpha=cutmix_alpha
                )
                buf_inputs = inputs.to(self.device)
                buf_labels_a = labels_a.to(self.device)
                buf_labels_b = labels_b.to(self.device)
                buf_outputs = self.net(buf_inputs)
                loss = lam * self.loss(buf_outputs, buf_labels_a) + (
                    1 - lam
                ) * self.loss(buf_outputs, buf_labels_b)
            else:
                buf_outputs = self.net(buf_inputs)
                loss = self.loss(buf_outputs, buf_labels)

            loss.backward()
            optimizer.step()


class GDumb(BaseModel):
    name = "gdumb"
    description = "Continual learning via GDumb."
    link = "https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf"

    def __init__(self, backbone, loss, lr, buffer_size, batch_size, gdumb, **_):
        super(GDumb, self).__init__(backbone, loss, lr)
        self.buffer = Buffer(buffer_size, self.device)
        self.task = 0
        self.gdumb_config = gdumb
        self.gdumb_config['batch_size'] = batch_size

    def observe(self, inputs, labels, not_aug_inputs):
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return 0

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        self.net.reset_parameters()
        fit_buffer(self, self.gdumb_config, self.buffer)
