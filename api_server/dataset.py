# coding=utf-8
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
import json
import pickle

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

class VtonDataset(data.Dataset):
    """
    仮想試着用データセットクラス
    """
    def __init__(self, args, root_dir, datamode = "train", pair_list_path = "train_pairs.csv" ):
        super(VtonDataset, self).__init__()
        self.args = args

        # RGB 画像に対する transform : [-1,1]
        self.transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
        #self.transform = transforms.Compose( [ transforms.ToTensor(), transforms.Lambda(lambda x: (x*2)-1.0) ] )

        # マスク画像に対する transform : [0,1]
        self.transform_mask = transforms.Compose( [ transforms.ToTensor(), ] )
        self.transform_mask_wNorm = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ] )

        self.image_height = self.args.image_height
        self.image_width = self.args.image_width
        self.dataset_dir = os.path.join( root_dir, datamode )

        self.cloth_names = []
        self.poseA_names = []
        self.poseB_names = []
        with open( pair_list_path, "r" ) as f:
            for line in f.readlines():
                names = line.strip().split(",")
                poseA_name = names[0]
                poseB_name = names[1]
                cloth_name = names[2]
                self.poseA_names.append(poseA_name)
                self.poseB_names.append(poseB_name)
                self.cloth_names.append(cloth_name)

        if( self.args.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "self.poseA_names[0:5] :", self.poseA_names[0:5])
            print( "self.poseB_names[0:5] :", self.poseB_names[0:5])
            print( "self.cloth_names[0:5] :", self.cloth_names[0:5])
            print( "len(self.poseA_names) :", len(self.poseA_names))
            print( "len(self.poseB_names) :", len(self.poseB_names))
            print( "len(self.cloth_names) :", len(self.cloth_names))

    def __len__(self):
        return len(self.cloth_names)

    def get_cloth_part( self, parsing_img, pose_tsr ):
        """
        人物パース画像から正解服（人物画像における服部分）のテンソルを取得
        """
        parsing_np = np.array(parsing_img)

        # 正解服
        cloth_pos = [5,6,7]
        cloth_mask = np.zeros(parsing_np.shape).astype(np.float32)
        for pos in cloth_pos:
            cloth_mask += (parsing_np == pos).astype(np.float32)

        cloth_mask_tsr = torch.from_numpy(cloth_mask)
        cloth_tsr = pose_tsr * cloth_mask_tsr + (1 - cloth_mask_tsr)
        cloth_mask_tsr = cloth_mask_tsr.view(1, self.image_height, self.image_width)
        return cloth_tsr, cloth_mask_tsr

    def get_body_shape( self, parsing_img ):
        """
        人物パース画像からダウンサンプリングでぼかした BodyShape のテンソルを取得
        """
        parsing_np = np.array(parsing_img)

        bodyshape_mask_np = (parsing_np > 0).astype(np.float32)
        bodyshape_mask_img = Image.fromarray((bodyshape_mask_np*255).astype(np.uint8))

        if( self.image_height == 256 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 4, self.image_height // 4), Image.BILINEAR)
        elif( self.image_height == 512 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 8, self.image_height // 8), Image.BILINEAR)
        elif( self.image_height == 1024 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 16, self.image_height // 16), Image.BILINEAR)
        else:
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 4, self.image_height // 4), Image.BILINEAR)
        """
        if( self.image_height == 256 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 16, self.image_height // 16), Image.BILINEAR)
        elif( self.image_height == 512 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 32, self.image_height // 32), Image.BILINEAR)
        elif( self.image_height == 1024 ):
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 64, self.image_height // 64), Image.BILINEAR)
        else:
            bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width // 16, self.image_height // 16), Image.BILINEAR)
        """
        bodyshape_mask_img = bodyshape_mask_img.resize((self.image_width, self.image_height), Image.BILINEAR)
        bodyshape_mask_tsr = self.transform_mask_wNorm(bodyshape_mask_img)
        return bodyshape_mask_tsr

    def get_agnotic( self, parsing_img, pose_tsr, agnostic_type ):
        """
        人物パース画像から agnotic 形状のテンソルを取得する
        """
        parsing_np = np.array(parsing_img)
        if( agnostic_type == "agnostic1" ):
            # 顔のみあり
            gmm_agnostic_pos = [1,2,4,13]
        elif( agnostic_type == "agnostic2" ):
            # 顔あり＋首なし＋腕（長袖の場合手のみ）あり＋下半身あり
            gmm_agnostic_pos = [1,2,3,4,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27] 
        elif( agnostic_type == "agnostic3" ):
            # 顔あり＋首なし＋下半身あり
            gmm_agnostic_pos = [1,2,4,8,9,12,13,16,17,18,19]
        else:
            gmm_agnostic_pos = [1,2,4,13]

        gmm_agnostic_mask = np.zeros(parsing_np.shape).astype(np.float32)
        for pos in gmm_agnostic_pos:
            gmm_agnostic_mask += (parsing_np == pos).astype(np.float32)

        gmm_agnostic_mask_tsr = torch.from_numpy(gmm_agnostic_mask)
        gmm_agnostic_tsr = pose_tsr * gmm_agnostic_mask_tsr - (1 - gmm_agnostic_mask_tsr)
        return gmm_agnostic_tsr

    def get_keypoints( self, pose_keypoints_dir, pose_name ):
        """
        姿勢情報の keypoints のテンソルを取得する
        """
        if( os.path.exists(os.path.join(self.dataset_dir, pose_keypoints_dir, pose_name.replace(".jpg",".png").replace(".png","_keypoints.json")) ) ):
            format = "json"
        elif( os.path.exists(os.path.join(self.dataset_dir, pose_keypoints_dir, pose_name.replace(".jpg",".png").replace(".png",".pkl"))) ):
            format = "pkl"
        else:
            format = "json"

        if( format == "pkl" ):
            # pkl ファイルから pose keypoints の座標値を取得
            keypoints_dat = - np.ones((18, 2), dtype=int) # keypoints の x,y 座標値
            with open(os.path.join(self.dataset_dir, pose_keypoints_dir, pose_name.replace(".jpg",".png").replace(".png",".pkl") ), 'rb') as f:
                pose_label = pickle.load(f)
                for i in range(18):
                    if pose_label['subset'][0, i] != -1:
                        keypoints_dat[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]

                keypoints_dat = np.asarray(keypoints_dat)
        else:
            with open(os.path.join(self.dataset_dir, pose_keypoints_dir, pose_name.replace(".jpg",".png").replace(".png","_keypoints.json") ), 'rb') as f:
                pose_label = json.load(f)
                keypoints_dat = pose_label['people'][0]['pose_keypoints_2d']
                #keypoints_dat = pose_label['people'][0]['pose_keypoints']
                keypoints_dat = np.array(keypoints_dat)
                keypoints_dat = keypoints_dat.reshape((-1,3))

        # ネットワークに concat して入力するための keypoints テンソルと visualation 用のテンソルを作成
        point_num = 18
        if( self.image_height == 256 ):
            r = 5
        elif( self.image_height == 512 ):
            r = 10
        elif( self.image_height == 1024 ):
            r = 20
        else:
            r = 5

        pose_keypoints_tsr = torch.zeros(point_num, self.image_height, self.image_width)   # ネットワークに concat して入力するための keypoints テンソル
        pose_keypoints_img = Image.new('L', (self.image_width, self.image_height))         # 画像としての keypoints
        pose_keypoints_img_draw = ImageDraw.Draw(pose_keypoints_img)                      # 
        for i in range(point_num):
            one_map = Image.new('L', (self.image_width, self.image_height))
            draw = ImageDraw.Draw(one_map)
            point_x = keypoints_dat[i, 0]
            point_y = keypoints_dat[i, 1]
            if( format == "pkl" ):
                point_x = point_x * self.image_width / 762
                point_y = point_y * self.image_height / 1000

            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_keypoints_img_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')

            one_map = self.transform_mask_wNorm(one_map)
            pose_keypoints_tsr[i] = one_map[0]

        pose_keypoints_img_tsr = self.transform_mask_wNorm(pose_keypoints_img)   # 画像としての keypoints のテンソル
        return pose_keypoints_tsr, pose_keypoints_img_tsr

    def get_tom_wuton_agnotic( self, parsing_img, pose_img ):
        """
        WUTON 形式の agnotic 形式のテンソルを取得する
        """
        parsing_np = np.array(parsing_img)
        tom_agnostic_wErase_pos = [5, 6, 7, 10]
        tom_agnostic_woErase_pos = [1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        pose_np = np.array(pose_img).astype(np.float32) / 255   # 0.0f ~ 1.0f

        # 背景
        pose_agnotic_bg_np = (parsing_np == 0).astype(np.float32)
        pose_agnotic_bg_np_RGB = np.zeros( (pose_agnotic_bg_np.shape[0], pose_agnotic_bg_np.shape[1], 3) ).astype(np.float32)
        pose_agnotic_bg_np_RGB[:,:,0], pose_agnotic_bg_np_RGB[:,:,1], pose_agnotic_bg_np_RGB[:,:,2] = pose_agnotic_bg_np, pose_agnotic_bg_np, pose_agnotic_bg_np
        pose_agnotic_np = (pose_np * pose_agnotic_bg_np_RGB)
        #save_image( torch.from_numpy(pose_agnotic_bg_np), "_debug/pose_agnotic_bg_np.png" )

        # 灰色以外の部分
        pose_agnotic_woErase_np = np.zeros(parsing_np.shape).astype(np.float32)
        for pos in tom_agnostic_woErase_pos:
            pose_agnotic_woErase_np += (parsing_np == pos).astype(np.float32)

        pose_agnotic_woErase_np_RGB = np.zeros( (pose_agnotic_woErase_np.shape[0], pose_agnotic_woErase_np.shape[1], 3) ).astype(np.float32)
        pose_agnotic_woErase_np_RGB[:,:,0], pose_agnotic_woErase_np_RGB[:,:,1], pose_agnotic_woErase_np_RGB[:,:,2] = pose_agnotic_woErase_np, pose_agnotic_woErase_np, pose_agnotic_woErase_np
        pose_agnotic_np += (pose_np * pose_agnotic_woErase_np_RGB)
        #save_image( torch.from_numpy(pose_agnotic_woErase_np), "_debug/pose_agnotic_woErase_np.png" )

        # 灰色部分
        pose_agnotic_wErase_np = np.zeros(parsing_np.shape).astype(np.float32)
        for pos in tom_agnostic_wErase_pos:
            pose_agnotic_wErase_np += (parsing_np == pos).astype(np.float32)

        kernel = np.ones((self.args.wuton_agnotic_kernel_size, self.args.wuton_agnotic_kernel_size), np.uint8)
        pose_agnotic_wErase_np = cv2.dilate(pose_agnotic_wErase_np, kernel)

        pose_agnotic_wErase_np_RGB = np.zeros( (pose_agnotic_wErase_np.shape[0],pose_agnotic_wErase_np.shape[1],3) ).astype(np.float32)
        pose_agnotic_wErase_np_RGB[:,:,0], pose_agnotic_wErase_np_RGB[:,:,1], pose_agnotic_wErase_np_RGB[:,:,2] = pose_agnotic_wErase_np, pose_agnotic_wErase_np, pose_agnotic_wErase_np

        # 灰色部分の貼り付け
        color = 100
        pose_agnotic_wErase_img = Image.fromarray(np.uint8(pose_agnotic_wErase_np_RGB*color), mode="RGB").convert('RGB')
        pose_agnotic_wErase_mask_img = Image.fromarray(np.uint8(pose_agnotic_wErase_np * 255), mode="L").convert("L")
        #pose_agnotic_wErase_img.save( "_debug/pose_agnotic_wErase_img.png" )
        #pose_agnotic_wErase_mask_img.save( "_debug/pose_agnotic_wErase_mask_img.png" )

        pose_agnotic_img = Image.fromarray(np.uint8(pose_agnotic_np*255), mode="RGB").convert('RGB')  # 0 ~ 255
        pose_agnotic_img = Image.composite(pose_agnotic_wErase_img, pose_agnotic_img, pose_agnotic_wErase_mask_img)

        # 顔部分の貼り付け
        face_pos = [1, 2, 4, 13]
        pose_agnotic_face_np = np.zeros(parsing_np.shape).astype(np.float32)
        for pos in face_pos:
            pose_agnotic_face_np += (parsing_np == pos).astype(np.float32)
        
        pose_agnotic_face_np_RGB = np.zeros( (pose_agnotic_face_np.shape[0],pose_agnotic_face_np.shape[1],3) ).astype(np.float32)
        pose_agnotic_face_np_RGB[:,:,0], pose_agnotic_face_np_RGB[:,:,1], pose_agnotic_face_np_RGB[:,:,2] = pose_agnotic_face_np, pose_agnotic_face_np, pose_agnotic_face_np
        pose_agnotic_face_np_RGB = ( pose_np * pose_agnotic_face_np_RGB ) * 255
        pose_agnotic_face_img = Image.fromarray(np.uint8(pose_agnotic_face_np_RGB), mode="RGB").convert('RGB')
        pose_agnotic_face_mask_img = Image.fromarray(np.uint8(pose_agnotic_face_np*255), mode="L").convert("L")
        pose_agnotic_img = Image.composite(pose_agnotic_face_img, pose_agnotic_img, pose_agnotic_face_mask_img)
        #pose_agnotic_img.save( "_debug/pose_agnotic_img.png" )

        # 反転画像
        pose_agnotic_woErase_mask_img = ImageOps.invert(pose_agnotic_wErase_mask_img)

        # Tensor 型に cast して正規化
        pose_wuton_agnotic_tsr = self.transform(pose_agnotic_img)
        pose_wuton_agnotic_woErase_mask_tsr = self.transform_mask(pose_agnotic_woErase_mask_img)
        return pose_wuton_agnotic_tsr, pose_wuton_agnotic_woErase_mask_tsr

    def __getitem__(self, index):
        cloth_name = self.cloth_names[index]
        poseA_name = self.poseA_names[index]
        poseB_name = self.poseB_names[index]

        #---------------------
        # cloth
        #---------------------
        cloth_img = Image.open( os.path.join(self.dataset_dir, "cloth", cloth_name) ).convert("RGB")
        cloth_tsr = self.transform(cloth_img)

        #---------------------
        # cloth mask
        #---------------------
        cloth_mask_img = Image.open( os.path.join(self.dataset_dir, "cloth_mask", cloth_name) ).convert("L")
        cloth_mask_tsr = self.transform_mask(cloth_mask_img)
        
        #---------------------
        # pose img
        #---------------------
        poseA_img = Image.open( os.path.join(self.dataset_dir, "poseA", poseA_name) ).convert("RGB")
        poseA_tsr = self.transform(poseA_img)
        poseB_img = Image.open( os.path.join(self.dataset_dir, "poseB", poseA_name) ).convert("RGB")
        poseB_tsr = self.transform(poseB_img)

        #---------------------
        # pose parsing
        #---------------------
        poseA_parsing_img = Image.open( os.path.join(self.dataset_dir, "poseA_parsing", poseA_name) ).convert("L")
        poseA_parsing_tsr = self.transform_mask(poseA_parsing_img)
        poseB_parsing_img = Image.open( os.path.join(self.dataset_dir, "poseB_parsing", poseA_name) ).convert("L")
        poseB_parsing_tsr = self.transform_mask(poseB_parsing_img)

        # 正解服
        poseA_cloth_tsr, poseA_cloth_mask_tsr = self.get_cloth_part( poseA_parsing_img, poseA_tsr )
        poseB_cloth_tsr, poseB_cloth_mask_tsr = self.get_cloth_part( poseB_parsing_img, poseB_tsr )

        # BodyShape
        poseA_bodyshape_mask_tsr = self.get_body_shape( poseA_parsing_img )
        poseB_bodyshape_mask_tsr = self.get_body_shape( poseB_parsing_img )

        # GMM agnostic の形状
        poseA_gmm_agnostic_tsr = self.get_agnotic( poseA_parsing_img, poseA_tsr, self.args.gmm_agnostic_type )
        poseB_gmm_agnostic_tsr = self.get_agnotic( poseB_parsing_img, poseB_tsr, self.args.gmm_agnostic_type )

        poseA_tom_agnostic_tsr = self.get_agnotic( poseA_parsing_img, poseA_tsr, self.args.tom_agnostic_type )
        poseB_tom_agnostic_tsr = self.get_agnotic( poseB_parsing_img, poseB_tsr, self.args.tom_agnostic_type )

        #---------------------
        # pose keypoints
        #---------------------
        poseA_keypoints_tsr, poseA_keypoints_img_tsr = self.get_keypoints( "poseA_keypoints", poseA_name )
        poseB_keypoints_tsr, poseB_keypoints_img_tsr = self.get_keypoints( "poseB_keypoints", poseB_name )

        #---------------------
        # pose human identity (TOM agnotic)
        #---------------------
        poseA_wuton_agnotic_tsr, poseA_wuton_agnotic_woErase_mask_tsr = self.get_tom_wuton_agnotic( poseA_parsing_img, poseA_img )
        poseB_wuton_agnotic_tsr, poseB_wuton_agnotic_woErase_mask_tsr = self.get_tom_wuton_agnotic( poseB_parsing_img, poseB_img )

        #---------------------
        # Grid image
        #---------------------
        grid_img = Image.open('grid.png')
        grid_tsr = self.transform(grid_img)

        results_dict = {
            "cloth_name" : cloth_name,                              # 服のファイル名
            "poseA_name" : poseA_name,                              # 人物画像（ポーズA）のファイル名
            "poseB_name" : poseB_name,                              # 人物画像（ポーズB）のファイル名
            "cloth_tsr" : cloth_tsr,                                # 服画像
            "cloth_mask_tsr" : cloth_mask_tsr,                      # 服マスク画像

            "poseA_tsr" : poseA_tsr,                                # 人物（ポーズA）の人物画像
            "poseA_parsing_tsr" : poseA_parsing_tsr,                # 人物（ポーズA）の人物パース画像
            "poseA_cloth_tsr" : poseA_cloth_tsr,                    # 人物（ポーズA）の正解服
            "poseA_cloth_mask_tsr" : poseA_cloth_mask_tsr,          # 人物（ポーズA）の正解服のマスク画像
            "poseA_bodyshape_mask_tsr" : poseA_bodyshape_mask_tsr,  # 人物（ポーズA）のダウンサンプリング後の BodyShape
            "poseA_gmm_agnostic_tsr" : poseA_gmm_agnostic_tsr,      # 人物（ポーズA）の GMM に入力する agnotic 形状
            "poseA_tom_agnostic_tsr" : poseA_tom_agnostic_tsr,      # 人物（ポーズA）の TOM に入力する agnotic 形状
            "poseA_keypoints_tsr" : poseA_keypoints_tsr,            # 人物（ポーズA）のネットワークに入力する keypoints 情報
            "poseA_keypoints_img_tsr" : poseA_keypoints_img_tsr,    # 人物（ポーズA）の表示用の keypoints 情報
            "poseA_wuton_agnotic_tsr" : poseA_wuton_agnotic_tsr,                           # 人物（ポーズA）の WUTON 形式の agnotic 画像
            "poseA_wuton_agnotic_woErase_mask_tsr" : poseA_wuton_agnotic_woErase_mask_tsr, # 人物（ポーズA）の WUTON 形式の agnotic 画像の灰色部分以外のマスク画像

            "poseB_tsr" : poseB_tsr,                                # 人物（ポーズB）の人物画像
            "poseB_parsing_tsr" : poseB_parsing_tsr,                # 人物（ポーズB）の人物パース画像
            "poseB_cloth_tsr" : poseB_cloth_tsr,                    # 人物（ポーズB）の正解服
            "poseB_cloth_mask_tsr" : poseB_cloth_mask_tsr,          # 人物（ポーズB）の正解服のマスク画像
            "poseB_bodyshape_mask_tsr" : poseB_bodyshape_mask_tsr,  # 人物（ポーズB）のダウンサンプリング後の BodyShape
            "poseB_gmm_agnostic_tsr" : poseB_gmm_agnostic_tsr,      # 人物（ポーズB）の GMM に入力する agnotic 形状
            "poseB_tom_agnostic_tsr" : poseB_tom_agnostic_tsr,      # 人物（ポーズA）の TOM に入力する agnotic 形状
            "poseB_keypoints_tsr" : poseB_keypoints_tsr,            # 人物（ポーズB）のネットワークに入力する keypoints 情報
            "poseB_keypoints_img_tsr" : poseB_keypoints_img_tsr,    # 人物（ポーズB）の表示用の keypoints 情報
            "poseB_wuton_agnotic_tsr" : poseB_wuton_agnotic_tsr,                           # 人物（ポーズB）の WUTON 形式の agnotic 画像
            "poseB_wuton_agnotic_woErase_mask_tsr" : poseB_wuton_agnotic_woErase_mask_tsr, # 人物（ポーズB）の WUTON 形式の agnotic 画像の灰色部分以外のマスク画像

            "grid_tsr" : grid_tsr,
        }
        return results_dict


class VtonDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True):
        super(VtonDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch