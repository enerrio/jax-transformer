# GPT-2 in Jax
---
This repo accompanies the [Equinox and Friends](https://enerrio.bearblog.dev/equinox-and-friends/) blog post and the [GPT-2 in Jax](https://enerrio.bearblog.dev/gpt2-in-jax/) blog post. Here's a description of each file:

* equinox_test.py: Code from `Equinox and Friends` blog post
* entry_point.py: Main entry script for running training and inference
* config.py: Hyperparameters for different GPT-2 model sizes
* run_train.py: Kicks off training of the model
* run_inference.py: Generates text using a pretrained model for a given prompt
* transformer/data.py: Sets up DataLoaders for model training
* transformer/model.py: GPT-2 model built with Jax and Equinox
* transformer/utils.py: Utility functions for model serialization and plotting
* transformer/train.py: Training loop
* transformer/infer.py: Runs inference on a pretrained model
* tests/: Unit tests
* the-verdict.txt: Small dataset for training
* war-and-peace.txt: Large dataset for training

This code was tested using:
* python==3.12.4
* jax==0.4.34
* jaxtyping==0.2.34
* optax==0.2.3
* equinox==0.11.7

`environment.yml` is an export of the conda environment used during development of this codebase. If you have conda installed on your machine and want to create create an identical environment with all the libraries ready to go, run this:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate jax
```

## Usage
---
To train the model:
```bash
python entry_point.py train --data the-verdict --nb_epochs 1 --batch_size 4 --plot_name train_loss.png --model_size small --experiment_name test01
```

To run inference on a model:
```bash
python entry_point.py infer --model_size small --model_name gpt2-small-test01.eqx --prompt "hello my dear, I am" --max_new_tokens 50
```
