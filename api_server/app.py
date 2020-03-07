# coding=utf-8
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import cv2
import requests
import random
from apex import amp

# flask
import flask
from flask_cors import CORS

# PyTorch
import torch
import torch.optim as optim

# 自作モジュール
from utils import conv_base64_to_pillow, conv_pillow_to_base64, conv_tensor_to_pillow, create_binary_mask
from utils import save_checkpoint, load_checkpoint
from networks import GMM, End2EndGenerator
from dataset import VtonDataset, VtonDataLoader

#======================
# グローバル変数
#======================
args = None

#-----------------
# flask 関連
#-----------------
app = flask.Flask(__name__)

# OPTIONS を受け取らないようにする（Access-Control-Allow-Origin エラー対策）
CORS(app, resources={r"*": {"origins": "*"}}, methods=['POST', 'GET'])

app.config['JSON_AS_ASCII'] = False     # 日本語文字化け対策
app.config["JSON_SORT_KEYS"] = False    # ソートをそのまま

#-----------------
# OpenPose 関連
#-----------------
#openpose_server_url = http://openpose_ubuntu_gpu_container:5010/openpose

#-----------------
# Graphonomy 関連
#-----------------
#graphonomy_server_url = http://graphonomy_server_gpu_container:5001/graphonomy

#-----------------
# 試着モデル関連
#-----------------
device = None
model_G = None
ds_test = None
dloader_test = None

#================================================================
# "http://host_ip:5000" リクエスト送信（＝ページにアクセス）時の処理
#================================================================
@app.route('/')
def index():
    return flask.render_template('index.html', title='virtual-try-on_webapi_flask') 

#================================================================
# "http://host_ip:5000/api_server" にリクエスト送信時の処理
#================================================================
#@app.route('/api_server', methods=['POST','OPTIONS'])
@app.route('/api_server', methods=['POST'])
def responce():
    print( "リクエスト受け取り" )
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    # ブラウザから送信された json データの取得
    if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
        json_data = json.loads(flask.request.json)
    else:
        json_data = flask.request.get_json()

    #
    poseA_img_path = os.path.join( args.dataset_dir, "test", "poseA", "1.png" )
    poseB_img_path = os.path.join( args.dataset_dir, "test", "poseB", "1.png" )
    cloth_img_path = os.path.join( args.dataset_dir, "test", "cloth", "1.png" )
    cloth_mask_img_path = os.path.join( args.dataset_dir, "test", "cloth_mask", "1.png" )
    poseA_keypoints_path = os.path.join( args.dataset_dir, "test", "poseA_keypoints", "1_keypoints.json" )
    poseB_keypoints_path = os.path.join( args.dataset_dir, "test", "poseB_keypoints", "1_keypoints.json" )

    #------------------------------------------
    # ブラウザから送信された画像データの変換
    #------------------------------------------
    pose_img_pillow = conv_base64_to_pillow( json_data["pose_img_base64"] )
    cloth_img_pillow = conv_base64_to_pillow( json_data["cloth_img_base64"] )
    pose_img_pillow = pose_img_pillow.resize( (args.image_width, args.image_height), resample = Image.LANCZOS )
    cloth_img_pillow = cloth_img_pillow.resize( (args.image_width, args.image_height), resample = Image.LANCZOS )
    pose_img_base64 = conv_pillow_to_base64( pose_img_pillow )
    cloth_img_base64 = conv_pillow_to_base64( cloth_img_pillow )

    pose_img_pillow.save( poseA_img_path )
    pose_img_pillow.save( poseB_img_path )
    cloth_img_pillow.save( cloth_img_path )

    #------------------------------------------
    # Graphonomy を用いて人物パース画像を生成する。
    #------------------------------------------
    graphonomy_msg = {'pose_img_base64': pose_img_base64 }
    graphonomy_msg = json.dumps(graphonomy_msg)     # dict を JSON 文字列として整形して出力
    try:
        graphonomy_responce = requests.post( args.graphonomy_server_url, json=graphonomy_msg )
        graphonomy_responce = graphonomy_responce.json()

    except Exception as e:
        print( "通信失敗 [Graphonomy]" )
        print( "Exception : ", e )
        #torch.cuda.empty_cache()

        http_status_code = 400
        response = flask.jsonify(
            {
                'status':'NG',
            }
        )
        return http_status_code, response

    pose_parse_img_base64 = graphonomy_responce["pose_parse_img_base64"]
    pose_parse_img_RGB_base64 = graphonomy_responce["pose_parse_img_RGB_base64"]
    pose_parse_img_pillow = conv_base64_to_pillow(pose_parse_img_base64)
    pose_parse_img_RGB_pillow = conv_base64_to_pillow(pose_parse_img_RGB_base64)

    pose_parse_img_pillow.save( os.path.join( args.dataset_dir, "test", "poseA_parsing", "1.png" ) )
    pose_parse_img_pillow.save( os.path.join( args.dataset_dir, "test", "poseB_parsing", "1.png" ) )
    pose_parse_img_RGB_pillow.save( os.path.join( args.dataset_dir, "test", "poseA_parsing", "1_vis.png" ) )
    pose_parse_img_RGB_pillow.save( os.path.join( args.dataset_dir, "test", "poseB_parsing", "1_vis.png" ) )

    #------------------------------------------
    # OpenPose を用いて人物姿勢 keypoints を生成する。
    #------------------------------------------
    # request 
    oepnpose_msg = {'pose_img_base64': pose_img_base64 }
    oepnpose_msg = json.dumps(oepnpose_msg)
    try:
        openpose_responce = requests.post(args.openpose_server_url, json=oepnpose_msg)
        openpose_responce = openpose_responce.json()
        with open( poseA_keypoints_path, 'w') as f:
            json.dump( openpose_responce, f, ensure_ascii=False )
        with open( poseB_keypoints_path, 'w') as f:
            json.dump( openpose_responce, f, ensure_ascii=False )

    except Exception as e:
        print( "通信失敗 [OpenPose]" )
        print( "Exception : ", e )
        #torch.cuda.empty_cache()

        http_status_code = 400
        response = flask.jsonify(
            {
                'status':'NG',
            }
        )
        return http_status_code, response

    #------------------------------------------
    # 前処理
    #------------------------------------------
    # 服マスク画像を生成する。
    cloth_mask_img_cv = create_binary_mask( cloth_img_path )
    cv2.imwrite( cloth_mask_img_path, cloth_mask_img_cv )

    #------------------------------------------
    # 試着モデルから試着画像を生成する。
    #------------------------------------------
    # ミニバッチデータを GPU へ転送
    inputs = dloader_test.next_batch()
    cloth_tsr = inputs["cloth_tsr"].to(device)
    cloth_mask_tsr = inputs["cloth_mask_tsr"].to(device)
    grid_tsr = inputs["grid_tsr"].to(device)

    poseA_tsr = inputs["poseA_tsr"].to(device)
    poseA_cloth_tsr = inputs["poseA_cloth_tsr"].to(device)
    poseA_cloth_mask_tsr = inputs["poseA_cloth_mask_tsr"].to(device)
    poseA_bodyshape_mask_tsr = inputs["poseA_bodyshape_mask_tsr"].to(device)
    poseA_gmm_agnostic_tsr = inputs["poseA_gmm_agnostic_tsr"].to(device)
    poseA_tom_agnostic_tsr = inputs["poseA_tom_agnostic_tsr"].to(device)
    poseA_keypoints_tsr = inputs["poseA_keypoints_tsr"].to(device)
    poseA_keypoints_img_tsr = inputs["poseA_keypoints_img_tsr"].to(device)
    poseA_wuton_agnotic_tsr = inputs["poseA_wuton_agnotic_tsr"].to(device)
    poseA_wuton_agnotic_woErase_mask_tsr = inputs["poseA_wuton_agnotic_woErase_mask_tsr"].to(device)

    poseB_tsr = inputs["poseB_tsr"].to(device)
    poseB_cloth_tsr = inputs["poseB_cloth_tsr"].to(device)
    poseB_cloth_mask_tsr = inputs["poseB_cloth_mask_tsr"].to(device)
    poseB_bodyshape_mask_tsr = inputs["poseB_bodyshape_mask_tsr"].to(device)
    poseB_gmm_agnostic_tsr = inputs["poseB_gmm_agnostic_tsr"].to(device)
    poseB_tom_agnostic_tsr = inputs["poseB_tom_agnostic_tsr"].to(device)
    poseB_keypoints_tsr = inputs["poseB_keypoints_tsr"].to(device)
    poseB_keypoints_img_tsr = inputs["poseB_keypoints_img_tsr"].to(device)
    poseB_wuton_agnotic_tsr = inputs["poseB_wuton_agnotic_tsr"].to(device)
    poseB_wuton_agnotic_woErase_mask_tsr = inputs["poseB_wuton_agnotic_woErase_mask_tsr"].to(device)

    with torch.no_grad():
        poseA_warp_cloth, poseA_warp_cloth_mask, poseA_warped_grid, \
        poseB_warp_cloth, poseB_warp_cloth_mask, poseB_warped_grid, \
        poseA_rough, poseA_attention, poseA_gen, \
        poseB_rough, poseB_attention, poseB_gen \
        = model_G( 
            cloth_tsr, cloth_mask_tsr, grid_tsr,
            poseA_tsr, poseA_bodyshape_mask_tsr, poseA_gmm_agnostic_tsr, poseA_tom_agnostic_tsr, poseA_keypoints_tsr, poseA_wuton_agnotic_tsr,
            poseB_tsr, poseB_bodyshape_mask_tsr, poseB_gmm_agnostic_tsr, poseB_tom_agnostic_tsr, poseB_keypoints_tsr, poseB_wuton_agnotic_tsr,
        )

        if( args.reuse_tom_wuton_agnotic ):
            poseA_gen = (1 - poseA_wuton_agnotic_woErase_mask_tsr) * poseA_gen + poseA_wuton_agnotic_woErase_mask_tsr * poseA_wuton_agnotic_tsr
            poseB_gen = (1 - poseB_wuton_agnotic_woErase_mask_tsr) * poseB_gen + poseB_wuton_agnotic_woErase_mask_tsr * poseB_wuton_agnotic_tsr


    tryon_img_pillow = conv_tensor_to_pillow( poseB_gen )
    tryon_img_base64 = conv_pillow_to_base64( tryon_img_pillow )

    #------------------------------------------
    # json 形式のレスポンスメッセージを作成
    #------------------------------------------
    #torch.cuda.empty_cache()
    http_status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            'tryon_img_base64': tryon_img_base64,
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
    #parser.add_argument('--host', type=str, default="localhost", help="IP アドレス / 0.0.0.0 でどこからでもアクセス可")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="IP アドレス / 0.0.0.0 でどこからでもアクセス可")
    parser.add_argument('--port', type=str, default="5000", help="ポート番号")
    parser.add_argument('--enable_threaded', action='store_true', help="並列処理有効化")

    parser.add_argument('--openpose_server_url', type=str, default="http://openpose_ubuntu_gpu_container:5010/openpose", help="OpenPose サーバーの URL")
    parser.add_argument('--graphonomy_server_url', type=str, default="http://graphonomy_server_gpu_container:5001/graphonomy", help="Graphonomy サーバーの URL")
    #parser.add_argument('--openpose_server_container_name', type=str, default="openpose_ubuntu_gpu_container", help="OpenPose サーバーのコンテナ名")
    #parser.add_argument('--openpose_server_port', type=str, default="5010", help="OpenPose サーバーのポート番号")
    #parser.add_argument('--graphonomy_server_container_name', type=str, default="graphonomy_server_gpu_container", help="OpenPose サーバーのコンテナ名")
    #parser.add_argument('--graphonomy_server_port', type=str, default="5001", help="OpenPose サーバーのポート番号")

    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="データセットのディレクトリ")
    parser.add_argument('--pair_list_path', type=str, default="datasets/test_pairs.csv", help="ペアリストファイル名")
    parser.add_argument('--load_checkpoints_gmm_path', type=str, default="checkpoints/improved_cp-vton_train_end2end_zalando_vton_dataset1_256_use_tom_agnotic_200215/GMM/gmm_final.pth", help="GMMモデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_tom_path', type=str, default="checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/TOM/tom_final.pth", help="TOMモデルの読み込みファイルのパス")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=256, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=192, help="入力画像の幅（pixel単位）")
    parser.add_argument('--grid_size', type=int, default=5)

    parser.add_argument('--poseA_bodyshape_downsampling_size', type=int, default=4)
    parser.add_argument('--poseB_bodyshape_downsampling_size', type=int, default=4)
    parser.add_argument('--gmm_agnostic_type', choices=['agnostic1', 'agnostic2', 'agnostic3'], default="agnostic1", help="GMM agnotic の形状タイプ")
    parser.add_argument('--tom_agnostic_type', choices=['agnostic1', 'agnostic2', 'agnostic3'], default="agnostic2", help="TOM agnotic の形状タイプ")
    parser.add_argument('--use_tom_wuton_agnotic', action='store_true', help="WUTON 形式の agnotic 入力を使用するか否か")
    parser.add_argument('--wuton_agnotic_kernel_size', type=int, default=6, help="WUTON 形式の agnotic 入力の膨張カーネルサイズ")
    parser.add_argument('--reuse_tom_wuton_agnotic', action='store_true', help="WUTON 形式の agnotic 入力を再利用するか否か")
    parser.add_argument('--eval_poseA_or_poseB', choices=['poseA', 'poseB'], default="poseB", help="人物画像の推論対象")

    parser.add_argument('--use_amp', action='store_true', help="AMP [Automatic Mixed Precision] の使用有効化")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument("--seed", type=int, default=8, help="乱数シード値")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    #------------------------------------
    # グローバル変数の設定
    #------------------------------------
    args = args
    #openpose_server_url = "http://" + args.openpose_server_container_name + ":" + args.openpose_server_port + "/openpose"
    #graphonomy_server_url = "http://" + args.graphonomy_server_container_name + ":" + args.graphonomy_server_port + "/graphonomy"

    #------------------------------------
    # 実行 Device の設定
    #------------------------------------
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            #torch.cuda.set_device(args.gpu_ids[0])
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #------------------------------------
    # 試着モデル
    #------------------------------------
    model_G = End2EndGenerator( args, device, use_cuda=True ).to(device)
    if not args.load_checkpoints_gmm_path == '' and os.path.exists(args.load_checkpoints_gmm_path):
        load_checkpoint( model_G.model_gmm, device, args.load_checkpoints_gmm_path )
    if not args.load_checkpoints_tom_path == '' and os.path.exists(args.load_checkpoints_tom_path):
        load_checkpoint( model_G.model_tom, device, args.load_checkpoints_tom_path )

    model_G.eval()

    #-------------------------------
    # AMP の適用（使用メモリ削減効果）
    #-------------------------------
    if( args.use_amp ):
        # dummy の optimizer
        optimizer = optim.Adam( params = model_G.parameters(), lr = 0.0001, betas = (0.5,0.999) )

        # amp initialize
        model_G, optimizer = amp.initialize(
            model_G, 
            optimizer, 
            opt_level = args.opt_level,
            num_losses = 1
        )

    #-------------------------------
    # DataLoader
    #-------------------------------
    ds_test = VtonDataset( args, args.dataset_dir, "test", args.pair_list_path )
    dloader_test = VtonDataLoader(ds_test, batch_size=args.batch_size, shuffle=False )

    #------------------------------------
    # Flask の起動
    #------------------------------------
    app.debug = args.debug
    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
