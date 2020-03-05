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

// 試着画像生成ボタンクリック時に呼び出される関数
function generateTryOnImage() {
    console.log( "試着画像の合成開始" );

    // 仮想試着サーバーの URL 取得
    var api_url = document.getElementById("api_url").value;

    // Canvas から画像データを取得
    var pose_img_canvas = document.getElementById("selected_file_pose_image_canvas");
    var pose_img_base64 = pose_img_canvas.toDataURL('image/png').replace(/^.*,/, '');
    var cloth_img_canvas = document.getElementById("selected_file_cloth_image_canvas");
    var cloth_img_base64 = cloth_img_canvas.toDataURL('image/png').replace(/^.*,/, '');

    // データを仮想試着サーバーに送信（jQuery での Ajax通信を開始）
    try {
        $.ajax({
            //url: 'http://0.0.0.0:5000/api_server',
            //url: 'http://localhost:5000/api_server',
            url: api_url,            
            type: 'POST',
            dataType: "json",
            data: JSON.stringify({ "pose_img_base64": pose_img_base64, "cloth_img_base64": cloth_img_base64 }),
            contentType: 'application/json',
            crossDomain: true,  // 仮想試着サーバーとリクエスト処理を異なるアプリケーションでデバッグするために必要
            timeout: 60000,
        })
        .done(function(data, textStatus, jqXHR) {
            // 通信成功時の処理を記述
            console.log( "試着画像の通信成功" );
            //console.log( data.tryon_img_base64 );
            console.log( textStatus );
            console.log( jqXHR );

            dataURL = `data:image/png;base64,${data.tryon_img_base64}`
            drawToCanvas( dataURL, "tryon_image_canvas" )
        })
        .fail(function(jqXHR, textStatus, errorThrown) {
            // 通信失敗時の処理を記述
            console.log( "試着画像の通信失敗" );
            console.log( textStatus );
            console.log( jqXHR );
            console.log( errorThrown );
        });
    } catch (e) {
        console.error(e)
        alert(e);
    }
}
