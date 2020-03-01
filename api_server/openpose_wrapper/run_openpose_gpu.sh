#!/bin/sh
IMAGE_DIR=../sample_n5
WRITE_JSON=../results_json
WRITE_IMAGE=../results_image

sudo mkdir -p ${IMAGE_DIR}
sudo mkdir -p ${WRITE_JSON}
sudo mkdir -p ${WRITE_IMAGE}

cd openpose_gpu
./build/examples/openpose/openpose.bin \
    --model_pose COCO \
    --image_dir ${IMAGE_DIR} --write_json ${WRITE_JSON} --write_images ${WRITE_IMAGE} \
    --display 0 \
    --hand

#./build/examples/openpose/openpose.bin \
#    --model_pose COCO \
#    --image_dir ${IMAGE_DIR} --write_json ${WRITE_JSON} \
#    --display 0 \
#    --render_pose 0 \
#    --hand