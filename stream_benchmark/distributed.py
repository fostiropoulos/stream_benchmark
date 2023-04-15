import argparse
import traceback
from pathlib import Path

import ray

from stream_benchmark.__main__ import train_method
from stream_benchmark.models import get_all_models

from stream_benchmark.utils.train import Logger

IGNORE_MODELS = []

INCLUDE_MODELS = []


def main(save_dir: Path, dataset_path: Path, num_cpus=0.001, num_gpus=0.15):
    logger = Logger(path=save_dir.joinpath("run.log"), verbose=True)
    model_names = get_all_models() if INCLUDE_MODELS is None else INCLUDE_MODELS
    model_names = set(model_names)
    model_names = model_names.difference(IGNORE_MODELS)
    remotes = []
    if len(IGNORE_MODELS) > 0:
        logger.warn(
            f"Ignoring models {IGNORE_MODELS} by default due to computational limitations. You can modify {__file__} to add the methods."
        )
    logger.info(f"Running in distributed mode for methods {model_names}")

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, max_calls=1)
    def remote_fn(model_name):
        try:
            train_method(
                save_path=save_dir,
                model_name=model_name,
                dataset_path=dataset_path,
                verbose=False,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error with {model_name}")

    for model_name in model_names:
        remotes.append(remote_fn.remote(model_name))
    ray.get(remotes)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", required=True, type=Path)
    args.add_argument("--save_dir", required=True, type=Path)
    args.add_argument(
        "--num_cpus",
        default=0.001,
        type=float,
        help="Fractional number of CPU cores to use. Set this to really small. It is not the bottleneck.",
    )
    args.add_argument(
        "--num_gpus",
        default=0.1,
        type=float,
        help="Fractional number of GPU to use. Set this so that {GPU usage per experiment} * num_gpus < 1",
    )
    kwargs = vars(args.parse_args())
    main(**kwargs)
