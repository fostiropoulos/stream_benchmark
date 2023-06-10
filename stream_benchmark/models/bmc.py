import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stream_benchmark.backbone.MLP import ResMLP
from stream_benchmark.models.__base_model import BaseModel
from stream_benchmark.utils.bmc import (
    Buffer,
    Memory,
    make_consolidation_dataloader,
    task_loss,
)
from stream_benchmark.utils.train import mixup_data, reset_optim_scheduler

from torch.optim import SGD, AdamW
from autods.main import AutoDS

from torch.optim.lr_scheduler import ReduceLROnPlateau


class BMC(BaseModel):
    name = "bmc"
    description = "Batch Model Consolidation"
    link = ""

    def __init__(self, backbone, loss, lr, minibatch_size, buffer_size, bmc, **_):
        super().__init__(backbone, loss, lr)
        self.task_coef = bmc["task_coef"]
        self.distil_coef = bmc["distil_coef"]
        self.consolidation_epochs = bmc["cons_epochs"]
        self.n_experts = bmc["n_experts"]
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.temp_buffers = []
        self.memory = Memory(bmc["memory_size"])
        self.base_model: ResMLP = copy.deepcopy(self.net)

    def _cons_forward(
        self,
        x,
        y,
        feats,
        task_offset,
        task_len,
    ):
        # if self.training:
        y = copy.deepcopy(y) + task_offset
        if self.base_model.training:
            x, y_a, y_b, lam = mixup_data(x, y, 0.4)
        else:
            y_a = y_b = y
            lam = 1
        backbone_logits, base_feats = self.base_model(x, "all")

        ce_loss_a = task_loss(
            logits=backbone_logits,
            y=y_a,
            task_offset=task_offset,
            task_len=task_len,
        )
        if lam < 1:
            ce_loss_b = task_loss(
                logits=backbone_logits,
                y=y_b,
                task_offset=task_offset,
                task_len=task_len,
            )
        else:
            ce_loss_b = 0
        ce_loss = lam * ce_loss_a + (1 - lam) * ce_loss_b

        loss = ce_loss * self.task_coef
        # if self.distil_coef > 0:
        #     loss += F.mse_loss(base_feats, feats).sum() * self.distil_coef

        return loss

    def consolidate(self):
        train_loader, test_loader = make_consolidation_dataloader(
            self.temp_buffers + self.memory.buffers, self.minibatch_size
        )

        optimizer = AdamW(self.base_model.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", threshold=1e-4, patience=10, verbose=False
        )
        for epoch in range(self.consolidation_epochs):
            self.base_model.train()
            for i, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = self._cons_forward(**batch)
                loss.backward()
                optimizer.step()
            self.base_model.eval()
            losses = []
            for i, batch in enumerate(test_loader):

                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    val_loss = self._cons_forward(**batch)
                    losses.append(val_loss.item())
            val_loss = np.mean(losses)
            scheduler.step(val_loss)

    def forward(self, x: torch.Tensor, in_task=False) -> torch.Tensor:
        # should be using base_model than net in evaluation
        if in_task:
            return self.net(x)
        else:
            return self.base_model(x)

    def end_task(self, train_loader: DataLoader, task_start_idx, *_):
        b = Buffer(
            self.net,
            train_loader,
            self.buffer_size,
            self.task_offset,
            self.task_idx,
            self.task_len,
        )
        self.temp_buffers.append(b)
        if self.task_idx % self.n_experts != 0 and self.task_idx != self.n_tasks:
            return

        self.consolidate()
        self.memory.extend(self.temp_buffers)
        self.temp_buffers = []

        return

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()

        logits, feats = self.net(inputs, "all")
        loss = (
            task_loss(
                logits=logits,
                y=labels,
                task_offset=torch.tensor(
                    [self.task_offset] * len(labels), device=logits.device
                ),
                task_len=torch.tensor(
                    [self.task_len] * len(labels), device=logits.device
                ),
            )
            * self.task_coef
        )

        # if self.distil_coef > 0:
        #     self.base_model.eval()
        #     with torch.no_grad():
        #         trainer_feats = self.base_model.features(inputs).detach()
        #     loss += F.mse_loss(feats, trainer_feats).sum() * self.distil_coef

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def begin_task(self, train_loader, task_start_idx):
        ds: AutoDS = train_loader.dataset

        self.task_len = ds.task_end_idxs[ds.task_id] - task_start_idx

        self.n_tasks = len(ds.task_end_idxs)
        self.task_offset = task_start_idx
        self.task_idx = ds.task_id

        state_dict = copy.deepcopy(self.base_model.state_dict())
        self.net.load_state_dict(state_dict=state_dict)
