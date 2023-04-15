# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stream_benchmark.models.__base_model import BaseModel


class Sgd(BaseModel):
    name = "sgd"
    description = "Stochastic gradient descent baseline without continual learning."
    link = "http://proceedings.mlr.press/v28/sutskever13.html"

    def __init__(self, backbone, loss, lr, **_):
        super().__init__(backbone, loss, lr)

    def end_task(self, *args, **kwargs):
        pass

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def begin_task(self, *_):
        self.net.reset_parameters()
