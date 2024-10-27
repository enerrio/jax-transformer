# GPT-2 in Jax
---
This repo accompanies the [Equinox and Friends](https://enerrio.bearblog.dev/equinox-and-friends/) blog post and the GPT-2 in Jax blog post (in progress). Here's a description of each file:

* equinox_test.py: Code from `Equinox and Friends` blog post

This code was tested using:
* jax==0.4.34
* jaxtyping==0.2.34
* optax==0.2.3
* equinox==0.11.7

> [!IMPORTANT]  
> Everything below is a work in progress for the `GPT-2 in Jax` blog post.

## Usage

To train the model:
> python entry_point.py train --nb_epochs 1 --batch_size 4 --plot_name train_loss.png --model_size small --experiment_name test01

To run inference on a model:
> python entry_point.py infer --model_size small --model_name gpt2-small-test01.eqx --prompt "hello my dear, I am" --max_new_tokens 50
