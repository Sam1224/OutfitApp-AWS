# coding=utf-8
import os
import io
import base64
import numpy as np
from PIL import Image
import cv2


def conv_base64_to_pillow( img_base64 ):
    decoded = base64.b64decode(img_base64)
    img_io = io.BytesIO(decoded)
    img_pillow = Image.open(img_io).convert('RGB')
    return img_pillow

def conv_base64_to_cv( img_base64 ):
    decoded = base64.b64decode(img_base64)
    img_np = np.fromstring(decoded, np.uint8)  
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img_cv

def conv_pillow_to_base64( img_pillow ):
    buff = io.BytesIO()
    img_pillow.save(buff, format="PNG")
    img_binary = buff.getvalue()
    img_base64 = base64.b64encode(img_binary).decode('utf-8')
    return img_base64
