import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file
from blind_watermark import WaterMark
import cv2
import numpy as np
import requests

app = Flask(__name__)
TEMP_DIR = "temp_server"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

TEXT_LEN = 8
REPEAT_COUNT = 5 

def process_image(img_path, text, out_path):
    """이미지 전처리 및 워터마크 삽입"""
    img = cv2.imread(img_path)
    
    # 메모리 보호: 너무 큰 이미지는 리사이징
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h, w = img.shape[:2]

    # 짝수 보정
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    if new_h != h or new_w != w:
        img = img[:new_h, :new_w]
    
    cv2.imwrite(img_path, img)

    # 비트 생성
    text = text.ljust(TEXT_LEN)[:TEXT_LEN]
    bits = []
    for char in text:
        bin_val = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in bin_val])
    wm_bits = bits * REPEAT_COUNT

    # 워터마크 삽입
    bwm = WaterMark(password_wm=1, password_img=1)
    bwm.read_img(img_path)
    bwm.read_wm(wm_bits, mode='bit')
    bwm.embed(out_path)

@app.route('/embed', methods=['POST'])
def embed():
    """기존: 파일 업로드 방식"""
    try:
        if 'image' not in request.files: return "No image", 400
        file = request.files['image']
        text = request.form.get('text', 'User1234')
        
        in_path = os.path.join(TEMP_DIR, "in.png")
        out_path = os.path.join(TEMP_DIR, "out.png")
        file.save(in_path)
        
        process_image(in_path, text, out_path)
        return send_file(out_path, mimetype='image/png', as_attachment=True, download_name='secured.png')
    except Exception as e:
        return str(e), 500

@app.route('/view', methods=['GET'])
def view():
    """신규: 링크 방식 (이게 있어야 사진이 보임!)"""
    try:
        file_id = request.args.get('id')
        text = request.args.get('text', 'Secure')
        
        if not file_id: return "No ID", 400

        # 구글 드라이브에서 다운로드
        url = f'https://drive.google.com/uc?export=view&id={file_id}'
        resp = requests.get(url)
        if resp.status_code != 200: return "Image not found", 404
        
        in_path = os.path.join(TEMP_DIR, f"in_{file_id}.png")
        out_path = os.path.join(TEMP_DIR, f"out_{file_id}.png")
        
        with open(in_path, 'wb') as f:
            f.write(resp.content)
            
        process_image(in_path, text, out_path)
        
        return send_file(out_path, mimetype='image/png')
        
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
