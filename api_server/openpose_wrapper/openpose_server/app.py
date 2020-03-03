# coding=utf-8
import os
import sys
import argparse
import json
from PIL import Image
import cv2
import numpy as np
import itertools

# flask
import flask
#from flask import Flask, render_template, request, jsonify

# openpose python API
sys.path.append('../openpose_gpu/build/python');
from openpose import pyopenpose as op

# 自作モジュール
from utils import conv_base64_to_pillow, conv_base64_to_cv, conv_pillow_to_base64

#======================
# グローバル変数
#======================
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

OPENPOSE_MODE_DIR_PATH = "../openpose_gpu/models/"

#================================================================
# "http://host_ip:5010" リクエスト送信時の処理
#================================================================
@app.route('/')
def index():
    print( "リクエスト受け取り" )
    return

#================================================================
# "http://host_ip:5010/openpose" にリクエスト送信時の処理
#================================================================
@app.route('/openpose', methods=['POST'])
def responce():
    print( "リクエスト受け取り" )
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    #------------------------------------------
    # 送信された json データの取得
    #------------------------------------------
    if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
        json_data = json.loads(flask.request.json)
    else:
        json_data = flask.request.get_json()

    #------------------------------------------
    # 送信された画像データの変換
    #------------------------------------------
    pose_img_cv = conv_base64_to_cv( json_data["pose_img_base64"] )
    if( app.debug ):
        cv2.imwrite( "tmp/pose_img.png", pose_img_cv )

    #------------------------------------------
    # OpenPose Python-API の実行
    # 参考 : openpose_gpu/build/examples/tutorial_api_python/01_body_from_image.py
    #------------------------------------------
    # パラメーターの設定
    params = dict()
    params["model_folder"] = OPENPOSE_MODE_DIR_PATH
    params["face"] = True
    params["hand"] = True

    # OpenPose Python-API
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    datum.cvInputData = pose_img_cv
    opWrapper.emplaceAndPop([datum])

    # keypoints の取得
    pose_keypoints_2d = np.delete( datum.poseKeypoints, [8, 19, 20, 21, 22, 23, 24], axis=1).reshape(-1).tolist()
    face_keypoints_2d = datum.faceKeypoints.reshape(-1).tolist()
    pose_keypoints_3d = datum.poseKeypoints3D.tolist()
    face_keypoints_3d = datum.faceKeypoints3D.tolist()
    left_hand_keypoints_2d = datum.handKeypoints[0].reshape(-1).tolist()
    right_hand_keypoints_2d = datum.handKeypoints[1].reshape(-1).tolist()
    hand_left_keypoints_3d = datum.handKeypoints3D[0].tolist()
    hand_right_keypoints_3d = datum.handKeypoints3D[1].tolist()
    """
    if( args.debug ):
        print("pose_keypoints_2d : ", pose_keypoints_2d )
        #print("pose_keypoints_2d[0][0] : ", pose_keypoints_2d[0][0] )
        #print("face_keypoints_2d: ", face_keypoints_2d )
        #print("pose_keypoints_3d: ", pose_keypoints_3d )
        #print("datum.cvOutputData: ", datum.cvOutputData )
    """

    #------------------------------------------
    # レスポンスメッセージの設定
    #------------------------------------------
    http_status_code = 200
    response = flask.jsonify(
        {
            "version" : 1.3,
            "people" : [
                {
                    "pose_keypoints_2d" : pose_keypoints_2d,
                    "face_keypoints_2d" : face_keypoints_2d,
                    "hand_left_keypoints_2d" : left_hand_keypoints_2d,
                    "hand_right_keypoints_2d" : right_hand_keypoints_2d,
                    "pose_keypoints_3d" : pose_keypoints_3d,
                    "face_keypoints_3d" : face_keypoints_3d,
                    "hand_left_keypoints_3d" : hand_left_keypoints_3d,
                    "hand_right_keypoints_3d" : hand_right_keypoints_3d,
                }
            ]
        }
    )

    # レスポンスメッセージにヘッダーを付与（Access-Control-Allow-Origin エラー対策）
    #response.headers.add('Access-Control-Allow-Origin', '*')
    #response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    if( app.debug ):
        print( "response.headers : \n", response.headers )

    return response, http_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--host', type=str, default="localhost", help="ホスト名（コンテナ名 or コンテナ ID）")
    #parser.add_argument('--host', type=str, default="openpose_ubuntu_gpu_container", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="ホスト名（コンテナ名 or コンテナ ID）")
    parser.add_argument('--port', type=str, default="5010", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
        
    if( args.debug ):
        app.debug = True
    else:
        app.debug = False

    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
