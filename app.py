import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file
from blind_watermark import WaterMark
import cv2
import numpy as np
import requests
import math

app = Flask(__name__)
TEMP_DIR = "temp_server"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# 🔐 설정: 타일링(쪼개기) 설정
TILE_SIZE = 400       # 타일 크기 (이 크기보다 작게 자르면 검출 불가)
TEXT_LEN = 8
REPEAT_COUNT = 3      # 타일 내 반복 횟수

def embed_tile(img_tile, text):
    """작은 타일 하나에 워터마크를 박는 함수"""
    try:
        h, w = img_tile.shape[:2]
        # 타일이 설정된 크기보다 작으면 패스 (가장자리 등)
        if h < TILE_SIZE or w < TILE_SIZE:
            return img_tile

        # 비트 생성
        text = text.ljust(TEXT_LEN)[:TEXT_LEN]
        bits = []
        for char in text:
            bin_val = bin(ord(char))[2:].zfill(8)
            bits.extend([int(b) for b in bin_val])
        wm_bits = bits * REPEAT_COUNT

        # 워터마크 삽입
        bwm = WaterMark(password_wm=1, password_img=1)
        
        # 라이브러리가 이미지를 읽게 하기 위해 임시 저장
        temp_tile_in = os.path.join(TEMP_DIR, "temp_tile_in.png")
        temp_tile_out = os.path.join(TEMP_DIR, "temp_tile_out.png")
        cv2.imwrite(temp_tile_in, img_tile)
        
        bwm.read_img(temp_tile_in)
        bwm.read_wm(wm_bits, mode='bit')
        bwm.embed(temp_tile_out)
        
        return cv2.imread(temp_tile_out)
    except:
        return img_tile # 에러나면 원본 타일 반환

def process_image_tiled(img_path, text, out_path):
    """전체 이미지를 쪼개서 처리하는 함수"""
    img = cv2.imread(img_path)
    
    # 1. 너무 큰 이미지는 리사이징 (속도 최적화)
    h, w = img.shape[:2]
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h, w = img.shape[:2]

    # 2. 패딩 추가 (이미지를 타일 크기의 배수로 맞춤)
    pad_h = (TILE_SIZE - (h % TILE_SIZE)) % TILE_SIZE
    pad_w = (TILE_SIZE - (w % TILE_SIZE)) % TILE_SIZE
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    padded_h, padded_w = img_padded.shape[:2]

    # 3. 루프 돌면서 타일마다 워터마크 박기
    for y in range(0, padded_h, TILE_SIZE):
        for x in range(0, padded_w, TILE_SIZE):
            # 타일 잘라내기
            tile = img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # 워터마크 박기
            watermarked_tile = embed_tile(tile, text)
            
            # 다시 붙이기
            img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE] = watermarked_tile

    # 4. 패딩 제거 (원래 크기로)
    final_img = img_padded[:h, :w]
    cv2.imwrite(out_path, final_img)

@app.route('/embed', methods=['POST'])
def embed():
    try:
        if 'image' not in request.files: return "No image", 400
        file = request.files['image']
        text = request.form.get('text', 'User1234')
        
        in_path = os.path.join(TEMP_DIR, "in_post.png")
        out_path = os.path.join(TEMP_DIR, "out_post.png")
        file.save(in_path)
        
        process_image_tiled(in_path, text, out_path)
        return send_file(out_path, mimetype='image/png', as_attachment=True, download_name='secured.png')
    except Exception as e:
        return str(e), 500

@app.route('/view', methods=['GET'])
def view():
    try:
        file_id = request.args.get('id')
        text = request.args.get('text', 'Secure')
        
        if not file_id: return "No ID", 400

        url = f'https://drive.google.com/uc?export=view&id={file_id}'
        resp = requests.get(url)
        if resp.status_code != 200: return "Image not found", 404
        
        in_path = os.path.join(TEMP_DIR, f"in_{file_id}.png")
        out_path = os.path.join(TEMP_DIR, f"out_{file_id}.png")
        
        with open(in_path, 'wb') as f:
            f.write(resp.content)
            
        process_image_tiled(in_path, text, out_path)
        
        return send_file(out_path, mimetype='image/png')
        
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
