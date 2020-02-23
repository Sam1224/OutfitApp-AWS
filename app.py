# coding=utf-8
import os
import argparse
import flask
#from flask import Flask, render_template, request, jsonify
import json
from utils import conv_base64_to_pillow, conv_pillow_to_base64

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

# "http://host_ip:5000" リクエスト送信（＝ページにアクセス）時の処理
@app.route('/')
def index():
    return flask.render_template('index.html', title='virtual-try-on_webapi_flask') 

# "http://host_ip:5000/try_on" にリクエスト送信時の処理
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

        pose_img_pillow = conv_base64_to_pillow( json_data["pose_img_base64"] )
        cloth_img_pillow = conv_base64_to_pillow( json_data["cloth_img_base64"] )
        if( app.debug ):
            pose_img_pillow.save("_debug/pose_img_pillow.png")
            cloth_img_pillow.save("_debug/cloth_img_pillow.png")

        tryon_img_pillow = pose_img_pillow.copy()
        tryon_img_base64 = conv_pillow_to_base64(tryon_img_pillow)

        # json 形式のレスポンス
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="localhost", help="IP アドレス / 0.0.0.0 でどこからでもアクセス可")
    parser.add_argument('--port', type=str, default="5000", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()

    if( args.debug ):
        if not os.path.exists("_debug"):
            os.mkdir("_debug")
        
    if( args.debug ):
        app.debug = True
    else:
        app.debug = False

    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
