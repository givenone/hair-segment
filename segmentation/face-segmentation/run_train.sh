#!/bin/bash

python main.py \
  --networks pspnet_resnet101 \
  --dataset lfw \
  --data_dir ./data/Lfw \
  --scheduler ReduceLROnPlateau \
  --batch_size 4 \
  --epochs 150 \
  --lr 1e-3 \
  --num_workers 2 \
  --optimizer adam \
  --img_size 256 \
  --momentum 0.5 \
  --ignite True
