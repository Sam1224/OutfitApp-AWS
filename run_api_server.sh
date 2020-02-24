#!/bin/sh
set -eu

python api_server/app.py \
    --host "localhost" --port 5000 \
    --debug
