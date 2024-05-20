#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=4 \
    $(dirname "$0")/train.py --launcher pytorch ${@:3}
