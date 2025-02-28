#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info nohup accelerate launch \
    --config_file accelerate_config/ds3.yaml \
    train_bt_pilot.py \
    recipes/samples/rm_bt.yaml 