#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --master_port 10015 --nproc_per_node=6 \
    $(dirname "$0")/test.py --zsd --launcher pytorch ${@:4}
