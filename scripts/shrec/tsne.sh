#!/usr/bin/env bash

## run the test and export collapses
python tsne.py \
--dataroot datasets/shrec_16 \
--name shrec16 \
--arch meshsimclr \
--dataset_mode simclr \
--ncf 64 128 256 512 \
--pool_res 600 450 300 180 \
--out_dim 512 \
--norm group \
--resblocks 5 \
--batch_size 36 \
--which_epoch latest \