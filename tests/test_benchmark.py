import copy
import io
import logging
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from stream.dataset import Dataset
from stream.main import Stream
from stream.utils import extract

from stream_benchmark.__main__ import train_method

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MockDataset(Dataset):
    metadata_url = "https://iordanis.xyz/"
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
            self.make_features(500, "cuda","clip")

    # ds.make_features(1024, DEVICE, clean=True, feature_extractor="clip")

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


class MockDataset2(MockDataset):
    name = "mock2"
    pass


class MockDataset3(MockDataset):
    name = "mock3"
    pass

class MockDataset4(MockDataset):
    name = "mock4"
    pass

def test_benchmark(tmp_path: Path):
    datasets = [MockDataset, MockDataset2, MockDataset3, MockDataset4]
    sizes = (np.arange(len(datasets)) + 1) * 100
    with mock.patch(
        "stream.main.Stream.supported_datasets",
        return_value=datasets,
    ):
        for ds, size in zip(datasets, sizes):
            ds(tmp_path, size=size, mock_download=True)

        with mock.patch(
            "stream.dataset.Dataset.assert_downloaded", return_value=True
        ), mock.patch("stream.dataset.Dataset.verify_downloaded", return_value=True):
            train_method(tmp_path, "sgd", tmp_path, "clip")
        breakpoint()
        return


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as fp:
        test_benchmark(Path(fp))
