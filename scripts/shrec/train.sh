#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/shrec_16 \
--name shrec16 \
--arch meshsimclr \
--dataset_mode simclr \
--ncf 64 128 256 512 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--niter_decay 100 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--rotate_and_shear \
--scale_verts \
--out_dim 512 \
--lr 0.0004 \
--batch_size 48 \
--num_threads 2 \