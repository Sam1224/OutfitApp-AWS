#!/bin/sh
set -e
CONTAINER_NAME=openpose_ubuntu_gpu_container

docker-compose -f docker-compose_gpu.yml stop
docker-compose -f docker-compose_gpu.yml up -d
docker exec -it -u $(id -u $USER):$(id -g $USER) ${CONTAINER_NAME} bash
