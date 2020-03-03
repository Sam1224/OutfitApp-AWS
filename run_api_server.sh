#!/bin/sh
set -eu
WORK_DIR=${PWD}

docker-compose -f docker-compose_gpu.yml stop
docker-compose -f docker-compose_gpu.yml up -d
#sleep 30

#-------------------------------
# Graphonomy の起動
#-------------------------------
if [ ! -e "${WORK_DIR}/api_server/graphonomy_wrapper/checkpoints/universal_trained.pth" ] ; then
    cd ${WORK_DIR}/api_server/graphonomy_wrapper
    sh download_model.sh
fi

#-------------------------------
# 仮想試着 API サーバーの起動
#-------------------------------
cd ${WORK_DIR}/api_server
python app.py \
    --host "0.0.0.0" --port 5000 \
    --debug
