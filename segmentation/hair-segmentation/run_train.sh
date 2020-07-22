#!/bin/bash

python main.py \
  --networks mobilenet \
  --dataset lfw \
  --data_dir ./data/Lfw \
  --scheduler ReduceLROnPlateau \
  --batch_size 4 \
  --epochs 100 \
  --lr 1e-3 \
  --num_workers 2 \
  --optimizer adam \
  --img_size 256 \
  --momentum 0.5 \
  --ignite True
