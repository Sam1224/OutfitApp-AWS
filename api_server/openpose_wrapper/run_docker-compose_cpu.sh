#!/bin/sh
set -e
CONTAINER_NAME=openpose_ubuntu_cpu_container

docker-compose -f docker-compose_cpu.yml stop
docker-compose -f docker-compose_cpu.yml up -d
docker exec -it -u $(id -u $USER):$(id -g $USER) ${CONTAINER_NAME} bash
