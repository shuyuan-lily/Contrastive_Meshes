#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/shrec_16 \
--name shrec16 \
--arch meshsimclr \
--dataset_mode simclr \
--ncf 64 128 256 512 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 5 \