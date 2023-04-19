
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from stream_benchmark.datasets import SequentialStream
from stream_benchmark.models.__base_model import BaseModel
from stream_benchmark.training import reset_optim_scheduler, timeit
from stream_benchmark.utils.train import Logger


def evaluate_joint(
    model: BaseModel, dataset: SequentialStream, cut_off_batch_n=None
) -> Tuple[List[float], List[float], List[float]]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, losses = [], []

    correct, total = 0.0, 0.0

    task_start_idx = torch.tensor(dataset.task_start_idx)
    dataset_lens = torch.tensor(dataset.dataset_len).cumsum(-1)

    test_loader = dataset.test_dataloader(shuffle = True)
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            (inputs, labels), index = data
            task_offset = torch.gather(
                task_start_idx, 0, (dataset_lens > index[:, None]).long().argmax(-1)
            )

            labels = task_offset + labels
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            losses.append(F.cross_entropy(outputs, labels).detach().cpu().numpy())
            pred = torch.argmax(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        if cut_off_batch_n is not None and i > cut_off_batch_n:
            break
        accs.append(correct / total * 100)
    val_loss = np.mean(losses)
    model.net.train(status)
    return accs, val_loss


def train_joint(
    model: BaseModel,
    dataset: SequentialStream,
    save_dir: Path,
    verbose: bool,
    n_epochs: int,
    early_stopping_patience: int,
    scheduler_patience: int,
    scheduler_threshold: float,
    **_,
) -> None:
    model.net.to(model.device)
    model.net.train()

    model.net.to(model.device)
    chkpt = save_dir.joinpath("results.pt")
    logger = Logger(path=save_dir.joinpath("train.log"), verbose=verbose)
    t = 0
    begin_task_duration = defaultdict(list)
    end_task_duration = defaultdict(list)
    observe_task_duration = defaultdict(lambda: defaultdict(list))

    train_loader = dataset.train_dataloader()
    task_name = train_loader.dataset.task_name
    task_num = train_loader.dataset.task_id
    task_start_idx = torch.tensor(dataset.task_start_idx)
    dataset_lens = torch.tensor(dataset.dataset_len).cumsum(-1)
    dl, duration = timeit(model.begin_task, train_loader, task_start_idx)
    if dl is not None:
        train_loader = dl

    begin_task_duration[t].append(duration)
    scheduler = reset_optim_scheduler(
        model, scheduler_patience, scheduler_threshold, verbose=verbose
    )
    val_loss = float("inf")
    mean_acc = 0
    running_loss = []
    loader = None
    if verbose:
        loader = tqdm(total=n_epochs * len(train_loader))
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            (inputs, labels), index = data
            task_offset = torch.gather(
                task_start_idx, 0, (dataset_lens > index[:, None]).long().argmax(-1)
            )

            labels = task_offset + labels
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            loss, duration = timeit(model.observe, inputs, labels, inputs)
            running_loss.append(loss)
            lr = model.optimizer.param_groups[0]["lr"]
            train_loss = np.mean(running_loss[-50:])
            if loader is not None:
                loader.set_description(
                    f"{task_name} - {epoch}/{n_epochs} loss: {train_loss:.4f} lr: {lr:5f} val_loss: {val_loss:4f} mean_acc: {mean_acc:.2f} "
                )
                loader.update(1)
            observe_task_duration[t][epoch].append(duration)
            if i % 100_000 == 0:
                accs, val_loss = evaluate_joint(model, dataset, cut_off_batch_n=10_000)
                mean_acc = np.mean(accs)
                logger.write_score(
                    mean_acc,
                    mean_acc,
                    val_loss,
                    task_name,
                    task_num,
                    prefix=f"epoch: {epoch}/{n_epochs} - sub-sample evaluation",
                )

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                    if scheduler.num_bad_epochs > early_stopping_patience:
                        logger.info(
                            f"Early stopping for task `{task_name}` ({task_num})"
                        )
                        break
        accs, val_loss = evaluate_joint(model, dataset, cut_off_batch_n=1_000)
        mean_acc = np.mean(accs)
        logger.write_score(
            mean_acc,
            mean_acc,
            val_loss,
            task_name,
            task_num,
            prefix=f"epoch: {epoch}/{n_epochs} - all-task evaluation",
        )
    _, duration = timeit(model.end_task, train_loader, task_start_idx)
    end_task_duration[t].append(duration)
