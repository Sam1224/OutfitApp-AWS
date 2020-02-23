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

