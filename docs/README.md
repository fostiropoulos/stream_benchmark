
<h1 style="text-align:center"> Batch Model Consolidation: A Multi-Task Model Consolidation Framework </h1>

<h3 style="text-align:center"> Iordanis Fostiropoulos &nbsp;&nbsp;&nbsp; Jiaye Zhu &nbsp;&nbsp;&nbsp; Laurent Itti</h3>
<p style="text-align:center"> University of Southern California</p>

<p style="text-align:center"> 
<a href="">[arXiv]</a> 
&nbsp;&nbsp;&nbsp; 
<a href="https://github.com/fostiropoulos/stream_benchmark">[Code]</a>  
&nbsp;&nbsp;&nbsp; 
<a href="https://github.com/fostiropoulos/stream">[Dataset]</a> 
</p>

## Abstract

In Continual Learning (CL), a model is required to learn a stream of tasks sequentially 
without significant performance degradation on previously learned tasks. 
Current approaches fail for a long sequence of tasks from diverse domains and difficulties. 
Many of the existing CL approaches are difficult to apply in practice due to excessive memory 
cost or training time, or are tightly coupled to a single device. With the intuition 
derived from the widely applied mini-batch training, we propose Batch Model Consolidation 
(**BMC**) to support more realistic CL under conditions where multiple agents are 
exposed to a range of tasks. During a _regularization_ phase, BMC trains multiple 
_expert models_ in parallel on a set of disjoint tasks. Each expert maintains weight 
similarity to a _base model_ through a _stability loss_, and constructs a 
_buffer_ from a fraction of the task's data. During the _consolidation_ phase, 
combine the learned knowledge on `batches' of _expert models_ using a 
_batched consolidation loss_ in _memory_ data that aggregates all buffers. 
We thoroughly evaluate each component of our method in an ablation study and demonstrate 
the effectiveness on standardized benchmark datasets Split-CIFAR-100, Tiny-ImageNet, 
and the Stream dataset composed of 71 image classification tasks from diverse domains 
and difficulties. Our method outperforms the next best CL approach by 70% and is the 
only approach that can maintain performance at the end of 71 tasks.

## Overview

Intuition: similar to mini-batch training, batched task training can reduce the local minima and improve the convexity of the loss landscape.

<p style="text-align:center">
<img src="https://drive.google.com/uc?export=view&id=1ZgwGy1Ta2u9Wim0D010uf7cSGw07qts9" alt="drawing" width="60%"/>
</p>

BMC optimizes multiple expert models from a single base model in parallel on **different** tasks,
enforcing parameter-isolation. Experts are regularized during training to reduce the forgetting 
on tasks learned by base model. A new base model is consolidated by **batched distillation** from the experts.

![A single incremental step of BMC](https://drive.google.com/uc?export=view&id=1nG4kD2PCP0sMZxBRD3LN8fZjzYvQrpTJ)

BMC supports distributed training where experts are trained locally on remote devices. 
Artifacts are sent back to the central device for consolidation training. 
The parallelism of this framework enables BMC to learn long task sequences efficiently.

![Paralleled multi-expert training framework](https://drive.google.com/uc?export=view&id=1NAswFVQtiNn6xkilUig42guGfvi-babV)

## The Stream Dataset

Stream dataset implements the logic for processing and managing a large sequence of datasets, 
and provides a method to train on interdisciplinary tasks by projecting all datasets on the same dimension,
by extracting features from pre-trained models.

See [the repository](https://github.com/fostiropoulos/stream/tree/cvpr_release) for Stream dataset installation and usages.

Download the extracted features for Stream datasets [here](https://drive.google.com/file/d/1insLK3FoGw-UEQUNnhzyxsql7z28lplZ/view).

## Class-Incremental Learning

We show on the Stream dataset with CLIP embedding that our method outperforms all other baselines in the Class-Incremental Learning scenario.
Our implementation of BMC as well as the baselines can be found [here](https://github.com/fostiropoulos/stream_benchmark).

![Experiment result on Stream](https://drive.google.com/uc?export=view&id=1rNjwxvOUYDcSOof9HTrD3eB0l0w_yM-8)



## Citation

```

```