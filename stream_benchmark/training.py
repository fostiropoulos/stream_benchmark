import time
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
from stream_benchmark.utils.train import Logger, mask_classes


def evaluate(
    model: BaseModel, dataset: SequentialStream, last=False
) -> Tuple[List[float], List[float], List[float]]:
    status = model.net.training
    model.net.eval()
    cil_acc, til_acc, losses = [], [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        start_idx = dataset.task_start_idx[k]
        for i, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data
                labels = labels + start_idx
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                losses.append(F.cross_entropy(outputs, labels).detach().cpu().numpy())
                pred = torch.argmax(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                mask_classes(outputs, dataset, k)
                pred = torch.argmax(outputs.data, 1)
                assert (labels >= dataset.task_start_idx[k]).all() and (
                    labels < dataset.task_end_idx[k]
                ).all()
                correct_mask_classes += torch.sum(pred == labels).item()

        cil_acc.append(correct / total * 100)
        til_acc.append(correct_mask_classes / total * 100)
    val_loss = np.mean(losses)
    model.net.train(status)
    return cil_acc, til_acc, val_loss


def timeit(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def reset_optim_scheduler(model: BaseModel, patience, threshold, verbose=True):
    model.reset_optim()
    return ReduceLROnPlateau(
        model.optimizer, "min", threshold=threshold, patience=patience, verbose=verbose
    )


def train(
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
    chkpt = save_dir.joinpath("results.pt")
    logger = Logger(path=save_dir.joinpath("train.log"), verbose=verbose)
    begin_task_duration = defaultdict(list)
    end_task_duration = defaultdict(list)
    observe_task_duration = defaultdict(lambda: defaultdict(list))
    cil_results = []
    til_results = []
    for t in range(dataset.task_id, dataset.n_tasks):
        model.net.train()
        train_loader = dataset.train_dataloader()

        task_name = train_loader.dataset.task_name
        task_num = train_loader.dataset.task_id
        task_start_idx = dataset.task_start_idx[task_num]
        dl, duration = timeit(model.begin_task, train_loader, task_start_idx)
        if dl is not None:
            train_loader = dl

        begin_task_duration[t].append(duration)
        scheduler = reset_optim_scheduler(
            model, scheduler_patience, scheduler_threshold, verbose=verbose
        )
        val_loss = float("inf")
        mean_cil_acc = 0
        mean_til_acc = 0
        running_loss = []
        loader = None
        if verbose:
            loader = tqdm(total=n_epochs * len(train_loader))
        for epoch in range(n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(train_loader.dataset, "logits"):
                    inputs, labels, logits = data
                    labels = task_start_idx + labels
                    inputs, labels, logits = (
                        inputs.to(model.device),
                        labels.to(model.device),
                        logits.to(model.device),
                    )
                    not_aug_inputs = inputs.clone().detach()
                    loss, duration = timeit(
                        model.observe, inputs, labels, not_aug_inputs, logits
                    )
                else:
                    inputs, labels = data
                    if model.name != "icarl":
                        labels = task_start_idx + labels
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = inputs.clone().detach()
                    loss, duration = timeit(model.observe, inputs, labels, inputs)
                running_loss.append(loss)
                lr = model.optimizer.param_groups[0]["lr"]
                train_loss = np.mean(running_loss[-50:])
                if loader is not None:
                    loader.set_description(
                        f"{task_name} - {epoch}/{n_epochs} loss: {train_loss:.4f} lr: {lr:5f} val_loss: {val_loss:4f} cil_acc: {mean_cil_acc:.2f} til_acc: {mean_til_acc:.2f}"
                    )
                    loader.update(1)
                observe_task_duration[t][epoch].append(duration)

            if model.name != "icarl" and model.name != "joint_gcl":
                cil_acc, til_acc, val_loss = evaluate(model, dataset, last=True)
                mean_cil_acc = np.mean(cil_acc)
                mean_til_acc = np.mean(til_acc)
                logger.write_score(
                    cil_acc,
                    til_acc,
                    val_loss,
                    task_name,
                    task_num,
                    prefix=f"epoch: {epoch}/{n_epochs} - in-task evaluation",
                )

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
                if scheduler.num_bad_epochs > early_stopping_patience:
                    logger.info(f"Early stopping for task `{task_name}` ({task_num})")
                    break

        _, duration = timeit(model.end_task, train_loader, task_start_idx)
        end_task_duration[t].append(duration)

        cil_acc, til_acc, val_loss = evaluate(model, dataset)
        mean_cil_acc = np.mean(cil_acc)
        mean_til_acc = np.mean(til_acc)
        cil_results.append(cil_acc)
        til_results.append(til_acc)
        torch.save(
            {
                "cil_results": cil_results,
                "til_results": til_results,
                "begin_task_duration": dict(begin_task_duration),
                "end_task_duration": dict(end_task_duration),
                "observe_task_duration": dict(observe_task_duration),
            },
            chkpt,
        )

        logger.write_score(
            cil_acc, til_acc, val_loss, task_name, task_num, prefix="all-tasks"
        )
        if t < dataset.n_tasks - 1:
            dataset.inc_task()
