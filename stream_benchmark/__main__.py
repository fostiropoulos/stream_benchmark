import argparse
import copy
import uuid
from pathlib import Path

import setproctitle
from stream_benchmark.datasets import SequentialStream
from stream_benchmark.models import get_model
from stream_benchmark.multi_task import train_joint
from stream_benchmark.training import train
import json


def train_method(
    save_path: Path,
    model_name: str,
    dataset_path: Path,
    hparams: Path,
    verbose=True,
):
    config = json.loads(hparams.read_text())
    batch_size = config["batch_size"]
    if model_name == "multi-task":
        dataset = SequentialStream(
            dataset_path,
            batch_size,
            task_id=None,
            num_workers=0,
        )
    else:
        dataset = SequentialStream(
            dataset_path,
            batch_size,
            task_id=0,
            num_workers=0,
        )
    verbose = verbose
    save_dir = save_path.joinpath(model_name, str(uuid.uuid4())[:4])
    save_dir.mkdir(parents=True)

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    task_offsets = dataset.task_start_idx
    config["dataset_path"] = dataset_path
    setproctitle.setproctitle(model_name)

    if model_name == "multi-task":
        model = get_model("sgd", backbone, loss, task_offsets, config)
        train_joint(model, dataset, save_dir, verbose, **config)
    else:
        model = get_model(model_name, backbone, loss, task_offsets, config)
        # set job name
        train(model, dataset, save_dir, verbose, **config)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--save_path", required=True, type=Path)
    args.add_argument("--model_name", required=True, type=str)
    args.add_argument("--dataset_path", required=True, type=Path)
    args.add_argument("--hparams", required=True, type=Path)
    kwargs = vars(args.parse_args())
    train_method(**kwargs)
