import copy
import io
import json
import logging
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from autods.dataset import Dataset
from autods.main import AutoDS
from autods.utils import extract
from PIL import Image

from stream_benchmark.__main__ import train_method
from stream_benchmark.datasets.seq_stream import include_ds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAKE_FEATS_BATCH_SIZE = 500
include_ds = ["mock1", "mock2", "mock3"]
hparams = {
    "early_stopping_patience": 10,
    "batch_size": 64,
    "buffer_size": 10000,
    "lr": 0.1,
    "minibatch_size": 64,
    "n_epochs": 20,
    "scheduler_threshold": 1e-4,
    "scheduler_patience": 10,
    "device": "cuda",
    "sgd": {},
}


class MockDataset(Dataset):
    metadata_url = "https://iordanis.me/"
    remote_urls = {"mock.tar": None}
    name = "mock"
    file_hash_map = {"mock.tar": "blahblah"}
    dataset_type = "image"
    default_task_name = "task1"

    task_names = ["task1", "task2", "task3"]

    def __init__(
        self, *args, mock_download=False, mock_process=True, size=100, **kwargs
    ) -> None:
        if mock_download:
            file_name = list(self.remote_urls)[0]
            archive_name, *_ = file_name.split(".")
            if len(args) > 0:
                root_path = args[0]
            else:
                root_path = kwargs["root_path"]
            archive_path = Path(root_path).joinpath(
                self.__class__.__name__.lower(), archive_name
            )
            rng = np.random.default_rng(seed=42)

            with tempfile.TemporaryDirectory() as fp:
                for split in ["train", "val"]:
                    for i in range(size):
                        img = Image.fromarray(
                            rng.integers(0, 255, (150, 150, 3)).astype(np.uint8)
                        )
                        name = f"{i}_{split}.png"
                        img.save(Path(fp).joinpath(name))
                shutil.make_archive(archive_path.as_posix(), "tar", root_dir=fp)
                self.file_hash_map[file_name] = self.file_hash(
                    str(archive_path) + ".tar"
                )
            if mock_process:
                kwargs["action"] = "process"
        super().__init__(*args, **kwargs)
        if mock_download:
            self.make_features(MAKE_FEATS_BATCH_SIZE, DEVICE, "clip")

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("mock.tar")
        extract(archive_path, raw_data_dir)

    def _make_metadata(self, raw_data_dir: Path):
        file_names = {}
        # NOTE if a dataset does not have subset variants. Simply add None as key
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                for f in raw_data_dir.joinpath("mock").glob(f"*_{split}.png"):
                    file_tuples.append(
                        (f.relative_to(raw_data_dir), np.random.randint(10))
                    )
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)


datasets = []
for ds in include_ds:

    class _MockClass(MockDataset):
        pass

    _MockClass.__name__ = ds.upper()
    _MockClass.name = ds
    datasets.append(_MockClass)

sizes = (np.arange(len(datasets)) + 1) * 100


def make_ds(self, task_id, train):

    with mock.patch(
        "autods.main.AutoDS.supported_datasets",
        return_value=datasets,
    ):

        transform = None
        if self.feats_name is None:
            transform = self.transforms(train)

        s = AutoDS(
            self.root_path,
            task_id=task_id,
            feats_name=self.feats_name,
            train=train,
            transform=transform,
            datasets=include_ds,
        )
    return s


def test_benchmark(tmp_path: Path):

    hpp = tmp_path.joinpath("hparams.json")
    hpp.write_text(json.dumps(hparams))

    for ds, size in zip(datasets, sizes):
        ds(tmp_path, size=size, mock_download=True)

    with mock.patch(
        "stream_benchmark.datasets.seq_stream.SequentialStream.make_ds", make_ds
    ), mock.patch(
        "autods.dataset.Dataset.assert_downloaded", return_value=True
    ), mock.patch(
        "autods.dataset.Dataset.verify_downloaded", return_value=True
    ):
        train_method(
            save_path=tmp_path, model_name="sgd", dataset_path=tmp_path, hparams=hpp
        )
    breakpoint()
    return


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as fp:
        test_benchmark(Path(fp))
