#!/bin/sh
set -eu
CONTAINER_NAME=openpose_ubuntu_gpu_container

docker-compose -f docker-compose_gpu.yml stop
docker-compose -f docker-compose_gpu.yml up -d
sleep 30

docker exec -it -u $(id -u $USER):$(id -g $USER) ${CONTAINER_NAME} /bin/bash -c "cd openpose_server && \
    python3 app.py \
        --host 0.0.0.0 --port 5010 \
        --debug"
