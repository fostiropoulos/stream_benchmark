
import torch.nn.functional as F
from stream.main import Stream
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms

from stream_benchmark.backbone.MLP import ResMLP
from stream_benchmark.backbone.resnet import ResNet


class SequentialStream:
    def __init__(
        self,
        root_path,
        batch_size,
        task_id=0,
        num_workers=0,
        feats_name=None,
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.task_id = task_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feats_name = feats_name
        self.image_size = 224
        self.val_image_size = 224
        mock_ds: Stream = self.make_ds(task_id, True)
        if isinstance(mock_ds.dataset, ConcatDataset):
            self.dataset_len = [len(ds) for ds in mock_ds.dataset.datasets]

        self.task_start_idx = [0] + list(mock_ds.task_end_idxs)
        self.task_end_idx = self.task_start_idx[1:]  # list(mock_ds.task_end_idxs)
        self.head_size = self.task_start_idx[-1]
        if self.feats_name is None:
            pass
        elif self.feats_name == "clip":
            self.feat_size = 768  # mock_ds[0][0].shape[0]
        elif self.feats_name == "vit":
            self.feat_size = 1024  # mock_ds[0][0].shape[0]
        elif self.feats_name == "resnet":
            self.feat_size = 2048  # mock_ds[0][0].shape[0]
        self.n_tasks = len(self.task_end_idx)
        self.test_loaders = [self.test_dataloader()]

    def transforms(self, train: bool):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(self.val_image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if train:
            return train_transform
        else:
            return test_transform

    def make_ds(self, task_id, train):
        transform = None
        if self.feats_name is None:
            transform = self.transforms(train)

        s = Stream(
            self.root_path,
            task_id=task_id,
            feats_name=self.feats_name,
            train=train,
            transform=transform,
            # process = True,
        )
        return s

    def make_dl(self, task_id, train, shuffle=True):
        ds = self.make_ds(task_id, train=train)
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 3
            kwargs["persistent_workers"] = True
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=train,
            **kwargs
        )
        return loader

    def train_dataloader(self):
        train_loader = self.make_dl(self.task_id, train=True)

        return train_loader

    def test_dataloader(self, shuffle=False):
        test_loader = self.make_dl(self.task_id, train=False, shuffle=shuffle)

        return test_loader

    def inc_task(self):
        self.task_id += 1
        self.test_loaders.append(self.test_dataloader())

    def get_backbone(self):
        if self.feats_name is not None:
            return ResMLP(self.feat_size, self.head_size)
        else:
            return ResNet(self.head_size)

    @staticmethod
    def get_loss():
        return F.cross_entropy