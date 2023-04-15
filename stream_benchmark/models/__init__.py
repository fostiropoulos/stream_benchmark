# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import importlib
from pathlib import Path
from copy import deepcopy
import pandas as pd
from stream_benchmark.models.__base_model import BaseModel



def get_all_models():
    model_dir = Path(__file__).parent.parent.joinpath("models").resolve().as_posix()

    return [
        model.split(".")[0]
        for model in os.listdir(model_dir)
        if not model.find("__") > -1 and "py" in model
    ]


MODELS = get_all_models()


def make_readme():

    table_rows = []
    for model_name in MODELS:
        mod = importlib.import_module("stream_benchmark.models." + model_name)
        mod_path = Path(mod.__file__)
        package_path = Path(__file__).parent.parent.parent
        model_class = get_model_class(model_name)
        table_rows += [
            {
                "description": model_class.description,
                "model_name": f"[{model_class.name}]({model_class.link})",
                "file": f"[{mod_path.name}]({mod_path.relative_to(package_path)})",
            }
        ]
    table = pd.DataFrame(table_rows).to_markdown(index=False)
    return table


def get_model_class(model_name) -> BaseModel:
    mod = importlib.import_module("stream_benchmark.models." + model_name)
    class_name = {x.lower(): x for x in mod.__dir__()}[model_name.replace("_", "")]
    return getattr(mod, class_name)


def get_model(model_name, backbone, loss, task_start_idx, configs):
    model_class = get_model_class(model_name)
    configs = deepcopy(configs)
    configs["task_start_idx"] = task_start_idx
    model = model_class(backbone, loss, **configs)
    model.to(configs["device"])
    return model


if __name__ == "__main__":
    print(make_readme())
