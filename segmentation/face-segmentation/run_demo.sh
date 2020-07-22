#!/bin/bash

python demo.py \
  --ckpt_dir ./ckpt/pspnet_resnet101_adam_lr_0.001_epoch_100.pth \
  --img_dir ./data/img \
  --networks pspnet_resnet101 \
  --save_dir ./overlay/ \
  --use_gpu True
