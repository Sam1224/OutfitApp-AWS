# 試着モデル関連
exper_name = "my-vton"
stage = "tom"
device = "gpu"
dataset_dir = "datasets"
pair_list_path = "datasets/test_pairs.csv"
load_checkpoints_gmm_path = "checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/GMM/gmm_final.pth"
load_checkpoints_tom_path = "checkpoints/my-vton_train_end2end_zalando_vton_dataset2_256_200218/TOM/tom_final.pth"
batch_size = 1
image_height = 256
image_width = 192
grid_size = 5

gmm_agnostic_type = "agnostic1"
tom_agnostic_type = "agnostic2"
use_tom_wuton_agnotic = False
wuton_agnotic_kernel_size = 6
reuse_tom_wuton_agnotic = True
eval_poseA_or_poseB = "poseB"

seed = 8
debug = True
