#!/usr/bin/env bash
python baselines/mdpo_ecpo_sac.py \
    --experiment "mdpo" \
    --env-name $1 \
    --seed $2 \
