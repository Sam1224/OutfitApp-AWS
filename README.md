# virtual-try-on_webapi_flask
Flask を用いた仮想試着モデルの WebAPI サーバー、及びブラウザアプリです。<br>
ブラウザアプリ上の GUI で指定した人物画像と服画像に対して、仮想試着サーバーで試着画像を生成します。<br>
仮想試着サーバーは、現在 GPU 版のみ動作します。（CPU 版は準備中）

使用可能な仮想試着モデルについては、以下のレポジトリを参照してくだい。
- [Yagami360/virtual-try-on_exercises_pytorch](https://github.com/Yagami360/virtual-try-on_exercises_pytorch)

## ■ 動作環境

### ◎ 仮想試着サーバー

- nvidia 製 GPU 搭載マシン : GPU 版のみ
- Ubuntu : GPU 版のみ
- docker : 
- docker-compose : 
- nvidia-docker2 : GPU 版のみ

### ◎ ブラウザアプリ

- Chrome
- Firefox

<!--
- javascript
    - jquery-3.4.1
-->

## ■ 使用法

1. 学習済み試着モデルのダウンロード<br>
    以下のスクリプトを実行し、仮想試着モデルの学習済みモデルをダウンロードします。
    ```sh
    $ sh download_models.sh
    ```

    その他のダウンロード可能な仮想試着モデルについては、以下のレポジトリを参照してくだい。
    - [Yagami360/virtual-try-on_exercises_pytorch](https://github.com/Yagami360/virtual-try-on_exercises_pytorch)

1. イメージの作成 & 仮想試着サーバーの起動<br>
    以下のコマンドを実行。docker イメージが作成されていない場合は、イメージの作成を行います。（イメージの作成には、時間がかかります。）<br>
    ```sh
    $ docker-compose stop
    $ docker-compose up -d
    ```

1. ブラウザアプリの起動
    1. `index.html` をブラウザで開く
    1. 画面上の GUI から、仮想試着サーバーの URL を設定する。
    1. 画面上の GUI から、人物画像と服画像のファイルを選択
    1. 「試着画像生成」ボタンをクリック 
