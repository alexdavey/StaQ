#!/usr/bin/env bash
# SAC baseline: MDPO policy update without the KL term
python baselines/mdpo_ecpo_sac.py \
    --experiment "sac" \
    --pupdate-type "SAC" \
    --env-name $1 \
    --seed $2 \
