#!/bin/bash

train_folder.py --root_dir ./structures/alignn_data \
    --epochs 20 --batch_size 4 \
    --config ./structures/alignn_data/config.json \
    --output_dir ./structures/alignn_output
