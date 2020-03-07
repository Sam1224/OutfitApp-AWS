$(function(){
    console.log("load page");
    var $pose_selectPanel = $('.pose_panel_select');
    var $cloth_selectPanel = $('.cloth_panel_select');

    //-------------------------------------------------
    // 変数 $pose_selectPanel, $cloth_selectPanel の要素をクリックしたとき
    //-------------------------------------------------
    $pose_selectPanel.on('click', function(e) {
        // その他の CSS の枠色をクリア
        $pose_selectPanel.css('border', '4px rgba(0,0,0,0) solid');
        $("#selected_file_pose_image_canvas").css('border', '4px rgba(0,0,0,0) solid')

        // クリックした要素のCSSを変更
        $(this).css('border', '4px blue solid');

        // Radio ボタンの選択を消す 
        document.getElementById('pose_select_0').checked = false;
        document.getElementById('pose_select_1').checked = false;
        document.getElementById('pose_select_2').checked = false;
        document.getElementById('pose_select_3').checked = false;
        document.getElementById('pose_select_4').checked = false;
        document.getElementById('pose_select_5').checked = false;

        console.log( this );
        console.log( this.children );
        console.log( this.children[0] );
        console.log( this.children[0].id );
        document.getElementById(this.children[0].id).checked = true;
    });
    $cloth_selectPanel.on('click', function(e) {
        $cloth_selectPanel.css('border', '4px rgba(0,0,0,0) solid');
        $("#selected_file_cloth_image_canvas").css('border', '4px rgba(0,0,0,0) solid')
        $(this).css('border', '4px blue solid');

        // Radio ボタンの選択を消す 
        document.getElementById('cloth_select_0').checked = false;
        document.getElementById('cloth_select_1').checked = false;
        document.getElementById('cloth_select_2').checked = false;
        document.getElementById('cloth_select_3').checked = false;
        document.getElementById('cloth_select_4').checked = false;
        document.getElementById('cloth_select_5').checked = false;
        document.getElementById(this.children[0].id).checked = true;
    });

    //-------------------------------------------------
    // 読み込み人物画像ファイル選択時に呼び出される関数（jQuery 使用）
    //-------------------------------------------------
    jQuery('#selected_file_pose_image').on('change', function(e) {
        // FileReader オブジェクトの作成
        var reader = new FileReader();
        reader.readAsDataURL(e.target.files[0]);    // ファイルが複数読み込まれた際に、1つ目を選択
        reader.onload = function (e) {  // 読み込みが成功時の処理
            img_src = e.target.result;
            drawToCanvas( img_src, "selected_file_pose_image_canvas" );
        }

        // 要素のCSSを変更
        $pose_selectPanel.css('border', '4px rgba(0,0,0,0) solid');
        $("#selected_file_pose_image_canvas").css('border', '4px blue solid');
    });

    //-------------------------------------------------
    // 読み込み服画像ファイル選択時に呼び出される関数（jQuery 使用）
    //-------------------------------------------------
    jQuery('#selected_file_cloth_image').on('change', function(e) {
        // FileReader オブジェクトの作成
        var reader = new FileReader();
        reader.readAsDataURL(e.target.files[0]);    // ファイルが複数読み込まれた際に、1つ目を選択
        reader.onload = function (e) {  // 読み込みが成功時の処理
            img_src = e.target.result;
            drawToCanvas( img_src, "selected_file_cloth_image_canvas" );
        }

        $cloth_selectPanel.css('border', '4px rgba(0,0,0,0) solid');
        $("#selected_file_cloth_image_canvas").css('border', '4px blue solid');
    });
});

//============================================
// 試着画像生成ボタンクリック時に呼び出される関数
//============================================
function generateTryOnImage() {
    console.log( "試着画像の合成開始" );

    // 仮想試着サーバーの URL 取得
    var api_url = document.getElementById("api_url").value;

    //---------------------------------------
    // 選択されている人物画像を取得
    //---------------------------------------
    radio_btn_pose0 = document.getElementById("pose_select_0");
    radio_btn_pose1 = document.getElementById("pose_select_1");
    radio_btn_pose2 = document.getElementById("pose_select_2");
    radio_btn_pose3 = document.getElementById("pose_select_3");
    radio_btn_pose4 = document.getElementById("pose_select_4");
    radio_btn_pose5 = document.getElementById("pose_select_5");
    console.log( "radio_btn_pose0.checked : ", radio_btn_pose0.checked );
    console.log( "radio_btn_pose1.checked : ", radio_btn_pose1.checked );
    console.log( "radio_btn_pose2.checked : ", radio_btn_pose2.checked );
    console.log( "radio_btn_pose3.checked : ", radio_btn_pose3.checked );
    console.log( "radio_btn_pose4.checked : ", radio_btn_pose4.checked );
    console.log( "radio_btn_pose5.checked : ", radio_btn_pose5.checked );

    var pose_img_base64
    if( radio_btn_pose0.checked == true ) {
        // Canvas から画像データを取得
        var pose_img_canvas = document.getElementById("selected_file_pose_image_canvas");
        pose_img_base64 = pose_img_canvas.toDataURL('image/png').replace(/^.*,/, '');
        //console.log( "pose_img_base64 : ", pose_img_base64 );
    }
    else if( radio_btn_pose1.checked == true ) {
        var pose_img = document.getElementById('pose_image_1');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_pose2.checked == true ) {
        var pose_img = document.getElementById('pose_image_2');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_pose3.checked == true ) {
        var pose_img = document.getElementById('pose_image_3');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_pose4.checked == true ) {
        var pose_img = document.getElementById('pose_image_4');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_pose5.checked == true ) {
        var pose_img = document.getElementById('pose_image_5');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }
    else{
        var pose_img = document.getElementById('pose_image_1');
        pose_img_base64 = convImageToBase64( pose_img, 'image/png' ).replace(/^.*,/, '');
    }

    //---------------------------------------
    // 選択されている服画像を取得
    //---------------------------------------
    radio_btn_cloth0 = document.getElementById("cloth_select_0");
    radio_btn_cloth1 = document.getElementById("cloth_select_1");
    radio_btn_cloth2 = document.getElementById("cloth_select_2");
    radio_btn_cloth3 = document.getElementById("cloth_select_3");
    radio_btn_cloth4 = document.getElementById("cloth_select_4");
    radio_btn_cloth5 = document.getElementById("cloth_select_5");
    console.log( "radio_btn_cloth0.checked : ", radio_btn_cloth0.checked );
    console.log( "radio_btn_cloth1.checked : ", radio_btn_cloth1.checked );
    console.log( "radio_btn_cloth2.checked : ", radio_btn_cloth2.checked );
    console.log( "radio_btn_cloth3.checked : ", radio_btn_cloth3.checked );
    console.log( "radio_btn_cloth4.checked : ", radio_btn_cloth4.checked );
    console.log( "radio_btn_cloth5.checked : ", radio_btn_cloth5.checked );

    var cloth_img_base64
    if( radio_btn_cloth0.checked == true ) {
        // Canvas から画像データを取得
        var cloth_img_canvas = document.getElementById("selected_file_cloth_image_canvas");
        cloth_img_base64 = cloth_img_canvas.toDataURL('image/png').replace(/^.*,/, '');
        //console.log( "cloth_img_base64 : ", cloth_img_base64 );
    }
    else if( radio_btn_cloth1.checked == true ) {
        var cloth_img = document.getElementById('cloth_image_1');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_cloth2.checked == true ) {
        var cloth_img = document.getElementById('cloth_image_2');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_cloth3.checked == true ) {
        var cloth_img = document.getElementById('cloth_image_3');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_cloth4.checked == true ) {
        var cloth_img = document.getElementById('cloth_image_4');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }
    else if( radio_btn_cloth5.checked == true ) {
        var cloth_img = document.getElementById('cloth_image_5');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }
    else{
        var cloth_img = document.getElementById('cloth_image_1');
        cloth_img_base64 = convImageToBase64( cloth_img, 'image/png' ).replace(/^.*,/, '');
    }

    //--------------------------------------------------------
    // データを仮想試着サーバーに送信（jQuery での Ajax通信を開始）
    //--------------------------------------------------------
    try {
        $.ajax({
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
            console.log( data.tryon_img_base64 );
            console.log( textStatus );
            console.log( jqXHR );

            dataURL = `data:image/png;base64,${data.tryon_img_base64}`
            drawToCanvas( dataURL, "tryon_image_canvas" )
        })
        .fail(function(jqXHR, textStatus, errorThrown) {
            // 通信失敗時の処理を記述
            console.log( "試着画像の通信失敗" );
            //console.log( textStatus );
            console.log( jqXHR );
            //console.log( errorThrown );
            alert("仮想試着サーバーとの通信に失敗しました\n" + api_url )
        });
    } catch (e) {
        console.error(e)
        alert(e);
    }
}
