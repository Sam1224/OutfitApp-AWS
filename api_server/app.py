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

# Self defined module
from utils import conv_base64_to_pillow, conv_pillow_to_base64, conv_tensor_to_pillow, create_binary_mask
from utils import save_checkpoint, load_checkpoint
from networks import GMM, End2EndGenerator
from dataset import VtonDataset, VtonDataLoader

#======================
# Global varaible
#======================
args = None

#-----------------
# flask
#-----------------
app = flask.Flask(__name__)

# Access-Control-Allow-Origin Policy
CORS(app, resources={r"*": {"origins": "*"}}, methods=['POST', 'GET'])

app.config['JSON_AS_ASCII'] = False
app.config["JSON_SORT_KEYS"] = False

#-----------------
# Virtual Try-on related variables
#-----------------
device = None
model_G = None
ds_test = None
dloader_test = None

#================================================================
# "http://host_ip:5000"
#================================================================
@app.route('/')
def index():
    return flask.render_template('index.html', title='virtual-try-on_webapi_flask')

#================================================================
# "http://host_ip:5000/api_server"
#================================================================
@app.route('/api_server', methods=['POST'])
def responce():
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
        json_data = json.loads(flask.request.json)
    else:
        json_data = flask.request.get_json()

    # file paths
    poseA_img_path = os.path.join( args.dataset_dir, "test", "poseA", "1.png" )
    poseB_img_path = os.path.join( args.dataset_dir, "test", "poseB", "1.png" )
    cloth_img_path = os.path.join( args.dataset_dir, "test", "cloth", "1.png" )
    cloth_mask_img_path = os.path.join( args.dataset_dir, "test", "cloth_mask", "1.png" )
    poseA_keypoints_path = os.path.join( args.dataset_dir, "test", "poseA_keypoints", "1_keypoints.json" )
    poseB_keypoints_path = os.path.join( args.dataset_dir, "test", "poseB_keypoints", "1_keypoints.json" )

    #------------------------------------------
    # imgs: base64 -> rgb format
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
    # Graphonomy: Human parsing (seperate each part, e.g. head, body, hand, etc.)
    #------------------------------------------
    graphonomy_msg = {'pose_img_base64': pose_img_base64 }
    graphonomy_msg = json.dumps(graphonomy_msg)
    try:
        graphonomy_responce = requests.post( args.graphonomy_server_url, json=graphonomy_msg )
        graphonomy_responce = graphonomy_responce.json()

    except Exception as e:
        print( "Fail to connect to the graphonomy server." )
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
    # OpenPose: Extract keypoints of human pose
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
        print( "Fail to connect to the openpose server." )
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
    # Preprocessing
    #------------------------------------------
    # Generate cloth mask
    cloth_mask_img_cv = create_binary_mask( cloth_img_path )
    cv2.imwrite( cloth_mask_img_path, cloth_mask_img_cv )

    #------------------------------------------
    # virtual try-on
    #------------------------------------------
    # Put a small batch of data onto gpu for calculating
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
    # Send generated image back in the format of base64
    #------------------------------------------
    #torch.cuda.empty_cache()
    http_status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            'tryon_img_base64': tryon_img_base64,
        }
    )

    # Cross-Domain Policy
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    if( app.debug ):
        print( "response.headers : \n", response.headers )

    return response, http_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="IP - 0.0.0.0 by default")
    parser.add_argument('--port', type=str, default="5000", help="port number")
    parser.add_argument('--enable_threaded', action='store_true', help="if multi-threaded")

    parser.add_argument('--openpose_server_url', type=str, default="http://openpose_ubuntu_gpu_container:5010/openpose", help="OpenPose Server URL")
    parser.add_argument('--graphonomy_server_url', type=str, default="http://graphonomy_server_gpu_container:5001/graphonomy", help="Graphonomy Server URL")

    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="use (CPU or GPU)")
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="dataset directory")
    parser.add_argument('--pair_list_path', type=str, default="datasets/test_pairs.csv", help="pair list file for testing")
    parser.add_argument('--load_checkpoints_gmm_path', type=str, default="checkpoints/improved_cp-vton_train_end2end_zalando_vton_dataset1_256_use_tom_agnotic_200215/GMM/gmm_final.pth", help="GMM model path")
    parser.add_argument('--load_checkpoints_tom_path', type=str, default="checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/TOM/tom_final.pth", help="TOM model path")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--image_height', type=int, default=256, help="image height")
    parser.add_argument('--image_width', type=int, default=192, help="image width")
    parser.add_argument('--grid_size', type=int, default=5)

    parser.add_argument('--poseA_bodyshape_downsampling_size', type=int, default=4)
    parser.add_argument('--poseB_bodyshape_downsampling_size', type=int, default=4)
    parser.add_argument('--gmm_agnostic_type', choices=['agnostic1', 'agnostic2', 'agnostic3'], default="agnostic1", help="GMM agnotic shape type")
    parser.add_argument('--tom_agnostic_type', choices=['agnostic1', 'agnostic2', 'agnostic3'], default="agnostic2", help="TOM agnotic shape type")
    parser.add_argument('--use_tom_wuton_agnotic', action='store_true', help="if use WUTON shape agnotic as input")
    parser.add_argument('--wuton_agnotic_kernel_size', type=int, default=6, help="increase kernel size for WUTON shape agnotic")
    parser.add_argument('--reuse_tom_wuton_agnotic', action='store_true', help="if reuse WUTON shape agnotic")
    parser.add_argument('--eval_poseA_or_poseB', choices=['poseA', 'poseB'], default="poseB", help="target")

    parser.add_argument('--use_amp', action='store_true', help="use AMP [Automatic Mixed Precision]")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument("--seed", type=int, default=8, help="random seed")
    parser.add_argument('--debug', action='store_true', help="if debug")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    #------------------------------------
    # Set global variable
    #------------------------------------
    args = args

    #------------------------------------
    # Set device: cpu/gpu
    #------------------------------------
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            print( "device :", device)
            print( "GPU :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "device :", device)
    else:
        device = torch.device( "cpu" )
        print( "device :", device)

    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #------------------------------------
    # Load Virtual Try-on model
    #------------------------------------
    model_G = End2EndGenerator( args, device, use_cuda=True ).to(device)
    if not args.load_checkpoints_gmm_path == '' and os.path.exists(args.load_checkpoints_gmm_path):
        load_checkpoint( model_G.model_gmm, device, args.load_checkpoints_gmm_path )
    if not args.load_checkpoints_tom_path == '' and os.path.exists(args.load_checkpoints_tom_path):
        load_checkpoint( model_G.model_tom, device, args.load_checkpoints_tom_path )

    model_G.eval()

    #-------------------------------
    # if use AMP (it can decrease the usage of memory)
    #-------------------------------
    if( args.use_amp ):
        # dummy optimizer
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
    # Start Flask
    #------------------------------------
    app.debug = args.debug
    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )
