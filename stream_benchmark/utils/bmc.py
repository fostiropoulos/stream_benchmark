import torch
from torch.utils.data.sampler import Sampler

import numpy as np
import pandas as pd
import bisect
import math
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torch.functional as F
import copy
from torch import nn
from stream_benchmark.backbone.MLP import ResMLP


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, labels: np.ndarray):
        super().__init__(data_source=None)
        self.weights = self._compute_weights(labels=labels)
        self._len = len(self.weights)

    @staticmethod
    def _compute_weights(labels: np.ndarray) -> np.ndarray:
        df = pd.DataFrame()
        df["label"] = labels
        counts = df["label"].value_counts() / len(labels)
        return 1.0 / counts[df["label"]].values

    def _compute_training_idxs(self) -> np.ndarray:
        sample_idx = np.arange(len(self.weights)).astype(int)
        probs = self.weights / self.weights.sum()
        idx = np.random.choice(
            sample_idx, size=len(self.weights), p=probs, replace=True
        ).tolist()
        return idx

    def __iter__(self):
        idx = self._compute_training_idxs()
        return iter(idx)

    def __len__(self):
        return self._len


def pair_wise_feat_loss(expert_feats, backbone_feats):
    layer_names = list(backbone_feats.keys())
    n = layer_names[-2]

    return F.mse_loss(expert_feats[n], backbone_feats[n]).sum()


def task_loss(logits, y, task_offset, task_len, reduction="mean"):
    # y_prime = copy.deepcopy(y)
    # this is mixed mask logits because it can include memory
    loss = 0
    if task_offset.max() != task_offset.min():
        end_idx = task_offset.max() + task_len[task_offset.argmax()]
        start_idx = task_offset.min()
        y_prime = y - start_idx
    else:
        _task_offset = task_offset[0].item()
        _task_len = task_len[0].item()
        start_idx = _task_offset
        end_idx = _task_offset + _task_len
        y_prime = copy.deepcopy(y) - _task_offset
    sliced_logits = logits[:, start_idx:end_idx]
    loss = nn.functional.cross_entropy(sliced_logits, y_prime, reduction=reduction)
    return loss


def mask_mixed_task_logits(
    logits,
    task_offset,
    task_len,
):
    # original author: http://juditacs.github.io/2018/12/27/masked-attention.html
    masked_logits = logits.clone()
    maxlen = masked_logits.size(1)
    idxs = torch.arange(maxlen, device=logits.device)[None, :]
    end_idxs = task_offset[:, None] + task_len[:, None]
    start_idxs = task_offset[:, None]
    mask_one = idxs >= start_idxs
    mask_two = idxs < end_idxs
    mask = mask_one & mask_two
    assert ((mask).sum(1) == task_len).all()
    return masked_logits[mask]


def make_consolidation_dataloader(buffers, batch_size):
    concat_datasets = ConcatDataset(buffers)

    # labels = np.arange(len(consolidation_dataset.datasets))
    labels = np.concatenate(
        [len(cons_ds.y) * [i] for i, cons_ds in enumerate(concat_datasets.datasets)]
    )
    val_size = math.ceil(len(concat_datasets) * 0.2)
    train_size = len(concat_datasets) - val_size
    consolidation_dataset, val_dataset = torch.utils.data.random_split(
        concat_datasets,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    cons_labels = labels[consolidation_dataset.indices]
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, sampler=None)

    balanced_sampler = ImbalancedDatasetSampler(labels=cons_labels)
    # NOTE shuffling happens inside the imbalanced dataset sampler
    cons_dataloader = DataLoader(
        consolidation_dataset,
        batch_size,
        shuffle=False,
        sampler=balanced_sampler,
        drop_last=True,
    )
    return cons_dataloader, val_loader


class Buffer(Dataset):
    def __init__(self, expert, train_loader, size, task_offset, task_idx, task_len):
        super().__init__()
        self.feats: List[np.array] = []
        self.x: List[np.array] = []
        self.y: List[np.array] = []
        self.task_offset = task_offset
        self.task_idx = task_idx
        self.task_len = task_len
        self.size = size
        self._make(expert, train_loader)

    def __getitem__(self, idx):
        task_offset = self.task_offset
        task_len = self.task_len
        x = self.x[idx]
        feats = self.feats[idx]
        y = self.y[idx]
        return {
            "x": x,
            "y": y,
            "feats": feats,
            "task_offset": task_offset,
            "task_len": task_len,
        }

    def __len__(self):
        return len(self.feats)

    def sample(self, n=None):
        if n is not None:
            self.size = n
        n = self.size
        idxs = np.random.permutation(len(self))[:n]
        self.feats = [self.feats[idx] for idx in idxs]
        self.x = [self.x[idx] for idx in idxs]
        self.y = [self.y[idx] for idx in idxs]

    @torch.no_grad()
    def _make(self, model: ResMLP, task_dataloader: DataLoader):
        model.eval()
        device = list(model.parameters())[0].device
        for i, batch in enumerate(task_dataloader):
            if len(batch) == 2:
                x, y = batch
            else:
                _, y, x = batch
            x = x.to(device)
            feats = model.features(x)
            for _x, _y, f in zip(
                x.detach().cpu().numpy(), y, feats.detach().cpu().numpy()
            ):
                self.feats.append(f)
                self.x.append(_x)
                self.y.append(_y)
                if len(self) > self.size:
                    break
            if len(self) > self.size:
                break


class Memory(Dataset):
    def __init__(self, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.buffers: List[Buffer] = []

    def sample(self) -> List[int]:
        slice_size = math.ceil(self.memory_size / len(self.buffers)) + 1
        [b.sample(slice_size) for b in self.buffers]

    def __len__(self):
        return np.sum([len(b) for b in self.buffers])

    def __getitem__(self, idx):
        row_counts = np.cumsum([len(b) for b in self.buffers]).astype(int)
        buffer_idx = bisect.bisect_right(row_counts, idx)
        if buffer_idx > 0:
            idx -= row_counts[buffer_idx - 1]
        return self.buffers[idx]

    def extend(self, buffers):
        if not (self.memory_size is None or self.memory_size > 0):
            return
        self.buffers += buffers
        self.sample()
