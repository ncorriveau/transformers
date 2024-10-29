# quick-trainer

Inspired by Andrej Karpathy, this repo serves as a flexible, configurable way 
to kick off LLM training experiments. All the configuration can be done 
in YAML, and the repo currently supports distributed CPU, single GPU, or multi-GPU training without any additional configuration needed. 

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

sh```
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

## Project Structure 
