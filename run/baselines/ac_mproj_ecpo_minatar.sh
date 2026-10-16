#!/usr/bin/env bash
python baselines/mdpo_ecpo_sac.py \
    --experiment "ecpo" \
    --pupdate-type "ECPO" \
    --env-name $1 \
    --seed $2 \
