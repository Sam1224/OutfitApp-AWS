# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
from tqdm import tqdm 
import requests

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64

# グローバル変数
IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="localhost", help="OpenPose サーバーのホスト名（コンテナ名 or コンテナ ID）")
    #parser.add_argument('--host', type=str, default="openpose_ubuntu_gpu_container", help="OpenPose サーバーのホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5010", help="OpenPose サーバーのポート番号")
    parser.add_argument('--image_dir', type=str, default="../sample_n5", help="入力静止画像のディレクトリ")
    parser.add_argument('--write_json', type=str, default="../results_json", help="json 形式の出力結果を保存するディレクトリ")
    parser.add_argument('--write_images', type=str, default="../results_image", help="画像形式の出力結果を保存するディレクトリ")
    parser.add_argument('--n_sample', type=int, default=1, help="OpenPose で処理する枚数")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    openpose_server_url = "http://" + args.host + ":" + args.port + "/openpose"
    #openpose_server_url = "http://" + args.host + ":" + args.port + "/"
    if( args.debug ):
        print( "openpose_server_url : ", openpose_server_url )

    image_names = sorted( [f for f in os.listdir(args.image_dir) if f.endswith(IMG_EXTENSIONS)] )
    for img_name in tqdm(image_names):
        #----------------------------------
        # リクエスト送信データの設定
        #----------------------------------
        pose_img_pillow = Image.open( os.path.join(args.image_dir, img_name) )
        pose_img_base64 = conv_pillow_to_base64(pose_img_pillow)

        #----------------------------------
        # リクエスト処理
        #----------------------------------
        oepnpose_msg = {'pose_img_base64': pose_img_base64 }
        oepnpose_msg = json.dumps(oepnpose_msg)
        openpose_responce = requests.post(openpose_server_url, json=oepnpose_msg)
        openpose_responce = openpose_responce.json()
        if( args.debug ):
            print( "openpose_responce : ", openpose_responce )

        #----------------------------------
        # ファイルに保存
        #----------------------------------
        