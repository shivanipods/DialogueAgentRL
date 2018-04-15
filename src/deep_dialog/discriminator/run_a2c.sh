#!/usr/bin/env bash
python A2C.py \
        --actor_lr 5e-4\
        --critic_lr 1e-3\
        --n 50\
        --gamma 1\
        --num-episodes 50000\
        --eval_after 500\
        --log_every 50\
