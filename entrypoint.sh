#!/usr/bin/env bash

python3 -m tensorboard.main --logdir=~/ray_results --host 0.0.0.0 &

exec "$@"