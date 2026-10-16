#!/usr/bin/env bash
python staq.py \
    --experiment "StaQ-minatar" \
    --rwd-scale 100.0 \
    --end-decay 1000000 \
    --hidden-width 128 \
    --env-name $1 \
    --seed $2 \
