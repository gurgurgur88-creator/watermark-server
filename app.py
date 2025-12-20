import os
# MKL 라이브러리 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file
from blind_watermark import WaterMark
import cv2
import numpy as np
import io

app = Flask(__name__)

# 임시 폴더 생성
TEMP_DIR = "temp_server"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# ★ 설정: 8글자 * 5번 반복 (압축/손상 방어용)
TEXT_LEN = 8
REPEAT_COUNT = 5 

def make_even(image):
    """비율 유지 + 짝수 크기 보정"""
    h, w, c = image.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    if new_h != h or new_w != w:
        return image[:new_h, :new_w]
    return image

def text_to_repeated_bits(text):
    """텍스트 -> 비트 변환 -> 5번 반복"""
    text = text.ljust(TEXT_LEN)[:TEXT_LEN]
    bits = []
    for char in text:
        bin_val = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in bin_val])
    return bits * REPEAT_COUNT

@app.route('/embed', methods=['POST'])
def embed():
    try:
        if 'image' not in request.files: return "No image", 400
        file = request.files['image']
        hidden_text = request.form.get('text', 'User1234')

        input_path = os.path.join(TEMP_DIR, "input.png")
        output_path = os.path.join(TEMP_DIR, "output.png")
        file.save(input_path)

        # 1. 이미지 읽기 및 짝수 보정
        img = cv2.imread(input_path)
        img = make_even(img)
        cv2.imwrite(input_path, img)

        # 2. 비트 생성
        wm_bits = text_to_repeated_bits(hidden_text)
        
        # 3. 워터마크 삽입
        bwm = WaterMark(password_wm=1, password_img=1)
        bwm.read_img(input_path)
        bwm.read_wm(wm_bits, mode='bit')
        bwm.embed(output_path)
        
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='secured.png')

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    # ★★★ [중요 수정] Render가 주는 포트를 받아야 합니다.
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)