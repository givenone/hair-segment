#!/bin/bash

python demo.py \
  --ckpt_dir ./ckpt/pspnet_resnet101_new_1.pth \
  --img_dir ./data/img \
  --networks pspnet_resnet101 \
  --save_dir ./overlay/ \
  --use_gpu True
