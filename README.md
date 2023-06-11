# Stream Benchmark

The code was used for the experiments and results of
**Batch-Model-Consolidation** [[arXiv]](https://openaccess.thecvf.com/content/CVPR2023/papers/Fostiropoulos_Batch_Model_Consolidation_A_Multi-Task_Model_Consolidation_Framework_CVPR_2023_paper.pdf) [[Website]](https://fostiropoulos.github.io/stream_benchmark/).
If using this code please cite:

```bibtex
@inproceedings{fostiropoulos2023batch,
  title={Batch Model Consolidation: A Multi-Task Model Consolidation Framework},
  author={Fostiropoulos, Iordanis and Zhu, Jiaye and Itti, Laurent},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3664--3676},
  year={2023}
}
```
This repository is a benchmark of methods found in [FACIL](https://github.com/mmasana/FACIL) and [Mammoth](https://github.com/aimagelab/mammoth) combined and adapted to work with the [AutoDS](https://github.com/fostiropoulos/auto-dataset) dataset to evaluate methods on a long sequence of tasks.



## Install

1. Install the [AutoDS dataset](https://github.com/fostiropoulos/auto-dataset).
2. `git clone https://github.com/fostiropoulos/stream_benchmark.git`
3. `cd stream_benchmark`
4. `pip install . stream_benchmark`


## AutoDS Feature Vectors [Download](https://drive.google.com/file/d/1insLK3FoGw-UEQUNnhzyxsql7z28lplZ/view)

We use 71 datasets with extracted features from pre-trained models,
supported in the AutoDS dataset. [The detailed table](https://github.com/fostiropoulos/auto-dataset/blob/cvpr_release/assets/DATASET_TABLE.md).

## Hyperparameters

Hyper-parameters are stored in [hparams/defaults.json](hparams/defaults.json)
with the reported values in their papers.
Modify the file for the number of `n_epochs` you want to train and the `batch_size` you want to use.

## Run a single method

```bash
python -m stream_benchmark --save_path {save_path} --dataset_path {dataset_path} --model_name {model_name} --hparams hparams/defaults.json
```

We run the baselines on Stream with [CLIP](https://arxiv.org/abs/2103.00020) embeddings in this code.
For `model_name` support see below.

## Run multiple methods in distribution

1. `ray stop`

2. `ray start --head`

3. `python -m stream_benchmark.distributed --dataset_path  {dataset_path} --num_gpus {num_gpus}`

**NOTE**:
`{num_gpus}` is the fractional number of GPU to use.
Set this so that `{GPU usage per experiment} * {num_gpus} < 1`

## Extending

The code in [test_benchmark.py](tests/test_benchmark.py) would be a good starting point in a simple example (ignoring the mock.patching) in understanding how the benchmark can be extended.


## Methods implemented
| Description                                                      | `model_name`                                                                                            | File                                                 |
| :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :--------------------------------------------------- |
| Continual learning via Gradient Episodic Memory.                 | [gem](https://arxiv.org/abs/1706.08840)                                                                 | [gem.py](stream_benchmark/models/gem.py)             |
| Continual learning via online EWC.                               | [ewc_on](https://arxiv.org/pdf/1805.06370.pdf)                                                          | [ewc_on.py](stream_benchmark/models/ewc_on.py)       |
| Continual learning via MAS.                                      | [mas](https://arxiv.org/abs/1711.09601)                                                                 | [mas.py](stream_benchmark/models/mas.py)             |
| Continual learning via Experience Replay.                        | [er](https://arxiv.org/abs/1811.11682)                                                                  | [er.py](stream_benchmark/models/er.py)               |
| Continual learning via Deep Model Consolidation.                 | [dmc](https://arxiv.org/abs/1903.07864)                                                                 | [dmc.py](stream_benchmark/models/dmc.py)             |
| Continual learning via A-GEM, leveraging a reservoir buffer.     | [agem_r](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | [agem_r.py](stream_benchmark/models/agem_r.py)       |
| Continual Learning Through Synaptic Intelligence.                | [si](https://arxiv.org/abs/1703.04200)                                                                  | [si.py](stream_benchmark/models/si.py)               |
| Continual learning via Function Distance Regularization.         | [fdr](https://arxiv.org/abs/1805.08289)                                                                 | [fdr.py](stream_benchmark/models/fdr.py)             |
| Gradient based sample selection for online continual learning    | [gss](https://arxiv.org/abs/1903.08671)                                                                 | [gss.py](stream_benchmark/models/gss.py)             |
| Continual learning via Dark Experience Replay++.                 | [derpp](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)  | [derpp.py](stream_benchmark/models/derpp.py)         |
| Continual learning via A-GEM.                                    | [agem](https://arxiv.org/abs/1812.00420)                                                                | [agem.py](stream_benchmark/models/agem.py)           |
| Stochastic gradient descent baseline without continual learning. | [sgd](http://proceedings.mlr.press/v28/sutskever13.html)                                                | [sgd.py](stream_benchmark/models/sgd.py)             |
| Continual learning via Learning without Forgetting.              | [lwf](https://arxiv.org/abs/1606.09282)                                                                 | [lwf.py](stream_benchmark/models/lwf.py)             |
| Continual Learning via iCaRL.                                    | [icarl](https://arxiv.org/abs/1611.07725)                                                               | [icarl.py](stream_benchmark/models/icarl.py)         |
| Continual learning via Dark Experience Replay.                   | [der](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)    | [der.py](stream_benchmark/models/der.py)             |
| Continual learning via GDumb.                                    | [gdumb](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf)                         | [gdumb.py](stream_benchmark/models/gdumb.py)         |
| Continual learning via Experience Replay.                        | [er_ace](https://arxiv.org/abs/1811.11682)                                                              | [er_ace.py](stream_benchmark/models/er_ace.py)       |
| Continual learning via Hindsight Anchor Learning.                | [hal](https://openreview.net/attachment?id=Hke12T4KPS&name=original_pdf)                                | [hal.py](stream_benchmark/models/hal.py)             |
| Joint training: a strong, simple baseline.                       | [joint_gcl]()                                                                                           | [joint_gcl.py](stream_benchmark/models/joint_gcl.py) |
