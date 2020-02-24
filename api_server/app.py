# coding=utf-8
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image

# flask
import flask
#from flask import Flask, render_template, request, jsonify

# PyTorch
import torch

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64

sys.path.append(os.path.join(os.getcwd(), 'graphonomy_wrapper'))
sys.path.append(os.path.join(os.getcwd(), 'graphonomy_wrapper/Graphonomy'))
from inference_all import inference
from networks import deeplab_xception_transfer, graph

#======================
# グローバル変数
#======================
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

model_graphonomy = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
    n_classes=20,
    hidden_layers=128,
    source_classes=7, 
).cpu()

model_graphonomy.load_state_dict( torch.load("graphonomy_wrapper/checkpoints/universal_trained.pth", map_location="cpu"), strict=False )

#================================================================
# "http://host_ip:5000" リクエスト送信（＝ページにアクセス）時の処理
#================================================================
@app.route('/')
def index():
    return flask.render_template('index.html', title='virtual-try-on_webapi_flask') 

#================================================================
# "http://host_ip:5000/try_on" にリクエスト送信時の処理
#================================================================
@app.route('/tryon', methods=['POST','OPTIONS'])
def responce():
    print( "リクエスト受け取り" )
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    if( flask.request.method == "POST" ):
        if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
            json_data = json.loads(flask.request.json)
        else:
            json_data = flask.request.get_json()

        #------------------------------------------
        # ブラウザから送信された画像データの変換
        #------------------------------------------
        pose_img_pillow = conv_base64_to_pillow( json_data["pose_img_base64"] )
        cloth_img_pillow = conv_base64_to_pillow( json_data["cloth_img_base64"] )
        pose_img_pillow.save("tmp/pose_img.png")
        cloth_img_pillow.save("tmp/cloth_img.png")

        #------------------------------------------
        # Graphonomy を用いて人物パース画像を生成する。
        #------------------------------------------
        pose_parse_img_pillow, pose_parse_img_RGB_pillow = get_humanparse_from_graphonomy( model_graphonomy, "tmp/pose_img.png", use_gpu=False )
        pose_parse_img_pillow.save("tmp/pose_parse_img.png")
        pose_parse_img_RGB_pillow.save("tmp/pose_parse_img_vis.png")

        #------------------------------------------
        # OpenPose を用いて人物姿勢 kepoints を生成する。
        #------------------------------------------

        #------------------------------------------
        # 試着モデルから試着画像を生成する。
        #------------------------------------------
        tryon_img_pillow = pose_parse_img_RGB_pillow.copy()
        tryon_img_base64 = conv_pillow_to_base64(tryon_img_pillow)

        #------------------------------------------
        # json 形式のレスポンスメッセージを作成
        #------------------------------------------
        http_status_code = 200
        response = flask.jsonify(
            {
                'status':'OK',
                'tryon_img_base64': tryon_img_base64,
            }
        )

    else:
        http_status_code = 200
        response = flask.jsonify(
            {
                'status':'OK',
            }
        )

    # レスポンスメッセージにヘッダーを付与（Access-Control-Allow-Origin エラー対策）
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    if( app.debug ):
        print( "response.headers : \n", response.headers )

    return response, http_status_code


def get_humanparse_from_graphonomy( model, pose_img_path, use_gpu=False ):
    pose_parse_img_np, pose_parse_img_RGB_pillow = inference( net=model, img_path=pose_img_path, use_gpu=use_gpu )
    pose_parse_img_pillow = Image.fromarray( np.uint8(pose_parse_img_np.transpose(0,1)) , 'L')
    return pose_parse_img_pillow, pose_parse_img_RGB_pillow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="localhost", help="IP アドレス / 0.0.0.0 でどこからでもアクセス可")
    parser.add_argument('--port', type=str, default="5000", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()

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
