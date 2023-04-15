# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stream_benchmark.utils.buffer import Buffer
from torch.nn import functional as F
from stream_benchmark.models.__base_model import BaseModel


class Der(BaseModel):
    name = "der"
    description = "Continual learning via Dark Experience Replay."
    link = "https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html"

    def __init__(self, backbone, loss, lr, der, buffer_size, minibatch_size, **_):
        super(Der, self).__init__(backbone, loss, lr)
        self.buffer = Buffer(buffer_size, self.device)
        self.minibatch_size = minibatch_size
        self.alpha = der['alpha']

    def begin_task(self, *_):
        pass

    def end_task(self, *_):
        pass

    def observe(self, inputs, labels, not_aug_inputs):

        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.minibatch_size
            )
            buf_outputs = self.net(buf_inputs)
            loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.optimizer.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
