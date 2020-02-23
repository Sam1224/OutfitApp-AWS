$(function(){
    console.log("load page");

    // 読み込み人物画像ファイル選択時に呼び出される関数（jQuery 使用）
    jQuery('#selected_file_pose_image').on('change', function(e) {
        // FileReader オブジェクトの作成
        var reader = new FileReader();
        reader.readAsDataURL(e.target.files[0]);    // ファイルが複数読み込まれた際に、1つ目を選択
        reader.onload = function (e) {  // 読み込みが成功時の処理
            img_src = e.target.result;
            drawToCanvas( img_src, "selected_file_pose_image_canvas" );
        }
    });

    // 読み込み服画像ファイル選択時に呼び出される関数（jQuery 使用）
    jQuery('#selected_file_cloth_image').on('change', function(e) {
        // FileReader オブジェクトの作成
        var reader = new FileReader();
        reader.readAsDataURL(e.target.files[0]);    // ファイルが複数読み込まれた際に、1つ目を選択
        reader.onload = function (e) {  // 読み込みが成功時の処理
            img_src = e.target.result;
            drawToCanvas( img_src, "selected_file_cloth_image_canvas" );
        }
	});
});

// canvas への画像表示を行う
function drawToCanvas( img_src, canvas_id ) {
    // Image オブジェクトの作成
    var img = new Image();
    img.src = img_src;
    img.onload = function(){
        // canvas に画像を表示
        var canvas = document.getElementById(canvas_id);
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        var context = canvas.getContext('2d');
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
}

// 試着画像生成ボタンクリック時に呼び出される関数
function generateTryOnImage() {
    console.log( "試着画像の合成開始" );

    // Canvas から画像データを取得
    var pose_canvas = document.getElementById("selected_file_pose_image_canvas");
    var pose_base64 = pose_canvas.toDataURL('image/png').replace(/^.*,/, '');
    var cloth_canvas = document.getElementById("selected_file_cloth_image_canvas");
    var cloth_base64 = cloth_canvas.toDataURL('image/png').replace(/^.*,/, '');

    // データを仮想試着サーバーに送信（jQuery での Ajax通信を開始）
    try {
        $.ajax({
            //url: 'http://0.0.0.0:5000/tryon',
            url: 'http://localhost:5000/tryon',
            type: 'GET',
            dataType: "json",
            data: JSON.stringify({ "pose_data": pose_base64, "cloth_data": cloth_base64 }),
            contentType: 'application/json',
            crossDomain: true,  // 仮想試着サーバーとリクエスト処理を異なるアプリケーションでデバッグするために必要
            timeout: 5000,
        })
        .done(function(data) {
            // 通信成功時の処理を記述
            console.log( "試着画像の通信成功" );
        })
        .fail(function() {
            // 通信失敗時の処理を記述
            console.log( "試着画像の通信失敗" );
        });
    } catch (e) {
        console.error(e)
        alert(e);
    }
}
