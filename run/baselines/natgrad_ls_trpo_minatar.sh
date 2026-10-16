#!/usr/bin/env bash
# Natural gradient with KL line search (--max-kl)
python baselines/nat_grad.py \
    --experiment "natgls" \
    --env-name $1 \
    --seed $2 \
