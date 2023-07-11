# RWKV Implementation for Infinite Context

This branch contains my experimental attempts to achieve infinite context training in RWKV.
With this implementation you can train on arbitrarily long context within (near) constant VRAM consumption; the increasing should be, take RWKV 7B as an example, about 2MB per 1024/2048 tokens (depending on your chosen `ctx_len`) in the training sample, which will enable training on sequences over 1M tokens.
Yet directly tune to such long sequences might be problematic; so `ctx_len_cutoff` is provided so longer sequences are sliced into multiple pieces of the specified cutoff size and learnt by the model separately.
It can be later increased until no cutoff presents.

The training code is by the way tremendously refactored into using PyTorch 2.0, Lightning 2.0 and DeepSpeed 2.0, and the starting script now relies on LightningCLI so you will see the [config.yaml](RWKV-v4neo/config-7B.yaml) containing all the switches, mostly standard ones that Lightning processes by itself.

To use this repo, go into `RWKV-v4neo` directory and do

```sh
python3 new_train.py fit -c {your_config}.yaml
```

Remember to modify the configuration for your own need. 

See [RWKV-v4neo/config-example.yaml](./RWKV-v4neo/config-example.yaml) for documentation on the various options

## Existing limitations

The following features are not yet supported (that may exist in [blinks original repo](https://github.com/BlinkDL/RWKV-LM))
- numpy file dataset
- binidx dataset
- model init weight
- model resize weights (init from smaller to bigger model)
- world tokenizer
- Learning Rate init -> Learning Rate Final support
- helper script to add new tokens to existing model

## Environment setup

The following venv setup using conda, modify for your use case respectively
```bash
# ninja-build is required for the new trainer
sudo apt-get install ninja-build

# Update conda & its package listings
conda update conda

# Virtual env, with python 3.10
# python 3.11 have issues with torch.compile / h100s
# and if you want to use 3.11, you will need to do a nightly build install
conda create -n rwkv-infctx python=3.11 pip
conda activate rwkv-infctx

# Install pytorch (>=2.0.1)
conda install -y pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Currently for torch.compile + 3.11 to work, for some paltforms, you will need the nightly build
# if so you may need to try the following instead
# ---
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia

# Verify your pytorch version 
python -c "import torch; print(torch.__version__)"

# We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python -m pip install datasets transformers 
python -m pip install lightning==2.0.4 deepspeed==0.9.5
python -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]'
python -m pip install lm-dataformat ftfy sentencepiece tokenizers wandb
```

Due to issues with [deepspeed on windows](https://github.com/microsoft/DeepSpeed/issues/2427). Only linux environments are supported. WSl2 with windows is not recommended, due to heavy performance penalities in the process (cannot use deepspeed offload, ~50% slower)

## Overall training process

- Either init a new model (todo script), or download an existing model
- Setup the [config.yaml](./RWKV-v4neo/config-example.yaml) file, customized for your foundation model / finetune use case
- Preload the dataset using the `python3 preload_dataset.py {you-config}.yaml`
- Start the training process `python3 new_train.py fit -c {your_config}.yaml`
- Export the checkpoint after training is complete with `python3 export_checkpoint.py ../path/to/checkpoint`
- From the checkpoint folder, you should find the fp32 model named `rwkv_model.pth`
- You should probably convert this to an fp16 model (todo script)

## Examples of dataset configs

@TODO