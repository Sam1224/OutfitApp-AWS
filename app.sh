#!/bin/sh
set -eu

python app.py \
    --host "localhost" --port 5000 \
    --debug
