# virtual-try-on_webapi_flask
Flask を用いた仮想試着モデルの WebAPI。<br>
ブラウザ上から指定した人物画像と服画像に対して、仮想試着モデルが生成した試着画像を表示します。

使用可能な仮想試着モデルについては、以下のレポジトリを参照してくだい。
- [Yagami360/virtual-try-on_exercises_pytorch](https://github.com/Yagami360/virtual-try-on_exercises_pytorch)

## ■ 動作環境（ブラウザ側）

- javascript
    - jquery-3.4.1

## ■ 動作環境（試着 API サーバー側）

- Ubuntu : 16.04
- Python : 3.6
- Anaconda
- PyTorch : 1.x 系
- tqdm
- Pillow : < 7.0.0
- flask : 
- flask-cors :
- networkx : Graphonomy 動作用

## ■ 使用法

