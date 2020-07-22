#!/bin/bash

python demo.py \
  --ckpt_dir ./ckpt/mobilenet_adam_lr_0.001_epoch_100.pth \
  --img_dir ./data/img \
  --networks mobilenet \
  --save_dir ./overlay/ \
  --use_gpu True
