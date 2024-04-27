#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python main.py \
--dataset 'MAFW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 1e-4 \
--weight-decay 1e-2 \
--print-freq 10 \
--temporal-layers 1 \
--img-size 224 \
--exper-name FINAL_224 \
