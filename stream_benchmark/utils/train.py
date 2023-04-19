from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import os
from stream_benchmark.datasets import SequentialStream
import random





def mask_classes(outputs: torch.Tensor, dataset: SequentialStream, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, : dataset.task_start_idx[k]] = -float("inf")
    outputs[:, dataset.task_start_idx[k + 1] :] = -float("inf")


class Logger:
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(
        self,
        path: str | Path | None = None,
        verbose: bool = True,
        prefix: str | None = None,
    ):
        self.path = path
        if path is not None:
            self.set_path(path)
        self.verbose = verbose
        self.set_prefix(prefix)

    def _write(self, msg: str):
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")

    def _print(self, msg: str, verbose=False):
        if self.verbose or verbose:
            print(msg)

    def info(self, msg: str, verbose=False):
        self(msg, verbose)

    def warn(self, msg: str, verbose=True):
        msg = f"{self.WARNING}{msg}{self.ENDC}"
        self(msg, verbose)

    def error(self, msg: str):
        msg = f"{self.FAIL}{msg}{self.ENDC}"
        self(msg, True)

    def __call__(self, msg: str, verbose=True):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{now}: {self.prefix}{msg}"
        self._write(msg)
        self._print(msg, verbose)

    def set_prefix(self, prefix: str | None = None):
        if prefix is not None:
            self.prefix = f"{prefix} - "
        else:
            self.prefix = ""

    def set_path(self, path: str | Path):
        self.path = Path(path)
        parent_dir = self.path.parent
        parent_dir.mkdir(exist_ok=True, parents=True)
        mode = "a" if os.path.exists(path) else "w"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(path, mode, encoding="utf-8") as f:
            f.write(f"Starting Logger {now} \n")


    def write_score(
        self,
        acc_class_il,
        acc_task_il,
        loss,
        task_name: str,
        task_num: int,
        prefix: str = "",
    ) -> None:
        """
        Prints the mean accuracy on stderr.
        :param mean_acc: mean accuracy value
        :param task_number: task index
        :param setting: the setting of the benchmark
        """
        mean_acc_class_il, mean_acc_task_il = np.mean(acc_class_il), np.mean(
            acc_task_il
        )
        msg = (
            f"{prefix} val-loss {loss:.2f} val-acc for task: {task_name} ({task_num}) - \t [Class-IL]: {round(mean_acc_class_il,2)} %"
            f" \t [Task-IL]: {round(mean_acc_task_il,2)} %"
        )
        self.info(msg, verbose=self.verbose)
