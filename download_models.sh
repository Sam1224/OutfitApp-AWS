#!/bin/sh
set -eu
mkdir -p api_server/checkpoints

#--------------
# GMM
#--------------
FILE_ID=1nX27sFpd-v-6_pehkEiqyGw47MZJavnH
FILE_NAME=gmm_final.pth

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
mkdir -p api_server/checkpoints/improved_cp-vton_train_end2end_zalando_vton_dataset1_256_use_tom_agnotic_200215/GMM
mv -i ${FILE_NAME} api_server/checkpoints/improved_cp-vton_train_end2end_zalando_vton_dataset1_256_use_tom_agnotic_200215/GMM

#--------------
# TOM
#--------------
FILE_ID=1xe7G-sTtqC_aVkTE0m4EMyTgmE-TaSzG
FILE_NAME=tom_final.pth

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
mkdir -p api_server/checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/TOM
mv -i ${FILE_NAME} api_server/checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/TOM

#--------------
# Graphonomy
#--------------
cd api_server/graphonomy_wrapper
sh download_model.sh
