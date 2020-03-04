#!/bin/sh
set -eu

docker-compose stop
docker-compose up -d

#docker exec -it -u $(id -u $USER):$(id -g $USER) api_server_gpu_container bash