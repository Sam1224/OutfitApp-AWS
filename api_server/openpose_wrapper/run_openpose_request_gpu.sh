#!/bin/sh
set -eu

#HOST_NAME=localhost
#HOST_NAME=openpose_ubuntu_gpu_container
HOST_NAME=0.0.0.0

cd openpose_server
python request.py \
    --host ${HOST_NAME} --port 5010 \
    --image_dir ../sample_n5 \
    --write_json ../results_json \
    --debug

#    --write_images ../results_image \