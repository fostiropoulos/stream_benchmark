from pathlib import Path

from stream.main import Stream
from torch.utils.data import DataLoader, Dataset


class AuxDataset(Dataset):
    def __init__(self, data, targets, logits=None):
        self.data = data
        self.targets = targets
        if logits is not None:
            self.logits = logits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, "logits"):
            return self.data[idx], self.targets[idx], self.logits[idx]
        else:
            return self.data[idx], self.targets[idx]


class AuxCIFAR100:
    def __init__(self, batch_size, dataset_path) -> None:
        self.dataset_path = dataset_path
        aux_train_ds, aux_test_ds = self.make_ds()
        self.train_loader = DataLoader(
            aux_train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            aux_test_ds,
            batch_size=batch_size,
            shuffle=False,
        )

    def make_ds(self):
        root_path = Path(self.dataset_path)

        # use the full dataset as aux data, no need to split
        train_ds = Stream(
            root_path=root_path, datasets=["cifar100"],  task_id = 0, feats_name="default", train=True
        )
        test_ds = Stream(
            root_path=root_path, datasets=["cifar100"], task_id = 0,  feats_name="default", train=False
        )
        return train_ds, test_ds


    def get_data_loaders(self):
        return self.train_loader, self.test_loader
