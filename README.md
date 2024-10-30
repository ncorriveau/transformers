# quick-trainer

Inspired by Andrej Karpathy, this repo serves as a flexible, configurable way 
to kick off LLM training experiments. The goal of the repo is to be able to take a new model card, update a YAML file for the particular components of that model (e.g. FFN dim, activation function, RoPE, order etc)
and then be able to train a real model from scratch with one command. It should work for single CPU, GPU, or multi GPU clusters out of the box, and this should largely be abstracted from the user outside of the distributed strategy to use. 

Currently the repo supports the following:
- ✔️ Sinusoidal Positional Encoding and RoPE
- ✔️ ReLU, GeLU, and SwiGLU activations
- ✔️ layer, batch and RMS norms
- ✔️ DDP and data parallel distributions
- ✔️ Checkpointing
- ✔️ Confirugable Transformer blocks - all in simple YAML!  

More to come! 

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Planned Work](#planned-work)

## Setup

The repo is set up with uv package manager, the necessary dependencies needed
to work within the repo are outlined in pyproject.toml 'dependencies' section. 

If you are using uv, you can set up a virtual environment and install the required libraries. 

```sh
uv venv

source .venv/bin/activate

uv pip install -e .
```

This should activate your virtual environment and install all the needed dependencies. 

## Usage 
The entry point for the repo is defined in src/train. There is a 
click CLI defined, or you can use the predefined script as outlined below: 

```sh
train --model-config-path ./configs/models/your_config.yaml  \
      --training-config-path ./configs/training/your_training_config.yaml \
      --data-path ./data/your_data
```

For convenience, the tiny shakespeare data set is already provided. Additionally, there is a default training config with parameters, and a model card config that defines the tiny GPT2 model. 

When you run this, the system will automatically figure out what compute you have available, and implement the distributed strategy if relevant. 

The configs/models/ directory should contain yaml defined decoder only LLM models. The interesting part here is that you can define not only the components that make up your transformer, but also the transformer sequencing itself by creating the transformer_block like so:

    transformer_block:
    - norm
    - attention
    - skip
    - norm
    - feed_forward
    - skip

This provides a very convenient wrapper to configuring your transformer block, to easily adapt to new variants with the existing components defined in the config as well. 

## Planned Work 
- FSDP support for parallel GPU training
- Support for datasets on S3
- Adding in UI support for tracking your experiments [Think a simplified W&B all local]
- Inference libraries for the pretrained models 

Reach out to me 📫 at ncorriveau13@gmail.com if you'd like to collaborate or contribute. 
