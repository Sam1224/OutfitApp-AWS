# graphonomy_wrapper
[Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) の推論スクリプト [`inference.py`](https://github.com/Gaoyiminggithub/Graphonomy/blob/master/exp/inference/inference.py) のラッパーモジュール。<br>
単一の画像ではなく、指定したフォルダ内の全人物画像に対して、人物パース画像を生成するように修正しています。

## ■ 動作環境
- Pytorch = 0.4.0 or Pytorch = 1.1.x
    - オリジナルの Graphonomy は Pytorch = 0.4.0 での動作環境になっているが、推論スクリプトは 1.x 系でも動作することを確認済み
- tqdm

## ■ 使い方
1. 以下のスクリプトを実装し、学習済みモデルをダウンロードして、checkpoints 以下のフォルダに保管する。
    ```sh
    $ sh download_model.sh
    ```

    - [Download 先 (Universal trained model)](https://drive.google.com/file/d/1sWJ54lCBFnzCNz5RTCGQmkVovkY9x8_D/view)<br>
    - 事前学習済みモデルの詳細は、オリジナルの [Graphonomy](ttps://github.com/Gaoyiminggithub/Graphonomy) の `README.md` を参照

1. 推論スクリプトを実行
    ```sh
    # 視覚用の人物パース画像の RGB画像も生成する場合
    # --in_image_dir : 入力人物画像のディレクトリ
    # --results_dir : 人物パース画像のディレクトリ
    $ python inference_all.py \
        --use_gpu 1 \
        --in_image_dir ${IN_IMAGE_DIR} \
        --results_dir ${RESULTS_DIR} \
        --load_checkpoints_path checkpoints/universal_trained.pth \
        --save_vis
    ```
    ```sh
    # グレースケール画像のみ生成する場合
    $ python inference_all.py \
        --use_gpu 1 \
        --in_image_dir ${IN_IMAGE_DIR} \
        --results_dir ${RESULTS_DIR} \
        --load_checkpoints_path checkpoints/universal_trained.pth
    ```
