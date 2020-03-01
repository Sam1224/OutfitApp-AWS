#!/bin/sh
set -eu

cd api_server
python app.py \
    --host "localhost" --port 5000 \
    --debug
