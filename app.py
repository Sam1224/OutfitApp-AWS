# coding=utf-8
import os
import argparse
#from flask import Flask, render_template, request, jsonify
import flask

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま


# Web 上の "http://host_ip:5000" ページにアクセス時の処理
# GET リクエスト : url に直接データを送る
# POST リクエスト : URLからは直接見れない形でデータが送る
@app.route('/')
def index():
    return flask.render_template('index.html', title='irtual-try-on_webapi_flask') 

@app.route('/responce', methods=['GET', 'POST'])
def responce():
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers : ", flask.request.headers )

    if( flask.request.method == "POST" ):
        data = flask.request.get_json()
        print( "data", data )
    
    # json 形式のレスポンス
    status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            "data" : data,
        }
    )

    return response, status_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="IP アドレス / 0.0.0.0 でどこからでもアクセス可")
    parser.add_argument('--port', type=str, default="5000", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()

    if( args.debug ):
        app.debug = True
    else:
        app.debug = False

    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
