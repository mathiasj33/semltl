[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Semantically Labelled Automata for Multi-Task Reinforcement Learning with LTL Instructions

This repository contains the official implementation of SemLTL (https://arxiv.org/pdf/2602.06746), as well as the environments *ConveyorWorld*, 
*LetterWorld*, and *ZoneEnv-8* (a modification of ZoneEnv with 8 distinct zones).

Also included are baselines [DeepLTL](https://arxiv.org/pdf/2410.04631) and
[LTL2Action](https://arxiv.org/pdf/2102.06858) for comparison. We rewrote these baselines to fit within our JAX implementation of SemLTL.

## Installation

We recommend using [pixi](https://pixi.sh/latest/) to install the required dependencies
in a virtual environment. Installing on GPU is highly recommended:
```bash
pixi install -e gpu
```

### SemML

The construction of our LDBAs and semantic features is implemented in the [SemML repository](https://gitlab.com/live-lab/software/semml), which is a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of this repository. Refer to [build.md](https://gitlab.com/live-lab/software/semml/-/blob/semerl/BUILD.md) for detailed building instructions. In short:
```bash
git submodule init
git submodule update
python3 semml/build.py
```

The python code expects SemML to be installed in a `dependencies` directory, so we create a symlink:

```bash
mkdir dependencies
ln -s semml dependencies/semml
```

To test if the SemML installation is working, the following command should output an LDBA:

```bash
python3 dependencies/semml/scripts_semml/embedd_ldba.py --formula="F a" --aps="a,b" --eligibleLetters="[[a],[b]]" --outputPath="tmp.hoa" && cat dependencies/semml/tmp.hoa && rm dependencies/semml/tmp.hoa
```

## Experiments

We use [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The below
commands assume you want to train and evaluate SemLTL on ZoneEnv-8.
You can set the experiment Hydra configuration via command line; see the
`conf` subfolder for the pre-made experiment configuration files.

### Precomputing Resets

For efficiency, we precompute the environment resets for both training and evaluation:
```bash
pixi run -e gpu python scripts/precompute_resets.py experiment=semltl/zones8 train=true
pixi run -e gpu python scripts/precompute_resets.py experiment=semltl/zones8 train=false
```

Repeat this for LetterWorld (this may take a few minutes):
```bash
pixi run -e gpu python scripts/precompute_resets.py experiment=semltl/letter train=true num_resets=1e5 rl_alg.num_envs=512
pixi run -e gpu python scripts/precompute_resets.py experiment=semltl/letter train=false num_resets=1e5 rl_alg.num_envs=512
```

Similarly, we precompute the training curriculum for faster training:
```bash
pixi run python scripts/precompute_curriculum.py experiment=semltl/zones8
```

> [!NOTE]
> Precomputing the curriculum will take a while to sample and build LDBAs. This will be much faster on subsequent runs.

### Training

To train a policy:
```bash
pixi run -e gpu python scripts/train.py experiment=semltl/zones8 run=first_run
```

To plot training performance:
```bash
pixi run -e gpu python scripts/plotting/plot_training_curves.py
```
**NOTE**: you will need to edit which runs to plot inside `scripts/plotting/plot_training_curves.py`

### Evaluation

To evaluate the trained policy on a set of LTL formulae:
```bash
pixi run -e gpu python scripts/eval/batch_eval.py experiment=semltl/zones8 run=first_run formulas=semltl/finite
```

> [!NOTE]
> We use max_steps_in_episode=5000 for the evaluation in ZoneEnv8. You will have to edit the `default_params` in `jaxltl/environments/zone_env8/zone_env8.py` to reproduce the numbers from the paper exactly.

We provide all formulae used in our experiments in `conf/formulas`. Depending on your hardware, you may run into memory issues when executing the evaluation script. If this is the case, you can pass `eval.formulas_per_batch=1` and `eval.models_per_batch=1` to reduce memory requirements at expense of longer runtime.

To visualize trajectories for the trained policy on an LTL formula (both for drawing trajectories and rendering them in real-time):
```bash
pixi run -e gpu python scripts/eval/visualize_trajectories.py experiment=semltl/zones8 run=first_run eval.formula="GF red & GF green"
```

## License

This project is licensed under the terms of the [MIT License](/LICENSE).