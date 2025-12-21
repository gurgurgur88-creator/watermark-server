import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file
from blind_watermark import WaterMark
import cv2
import numpy as np
import shutil

app = Flask(__name__)
TEMP_DIR = "temp_server"

if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR)

# ==========================================
# 🔐 [설정] TO.000000 ~ TO.999999 대응
# ==========================================
TILE_SIZE = 200       
PWD_IMG = 1
PWD_WM  = 1234
FIXED_BYTE_LEN = 9  # 🔥 [수정] 8 -> 9 (9글자로 늘림)
# ==========================================

def text_to_bits_fixed(text: str, fixed_len_bytes: int = 9):
    # 입력 텍스트를 9글자(72비트)로 강제 고정
    # 짧으면 뒤에 공백이 붙고, 길면 9글자에서 자름
    s = (text[:fixed_len_bytes]).ljust(fixed_len_bytes)
    bits = []
    for ch in s:
        b = ord(ch)
        bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
    return bits

def embed_tile(img_tile, text):
    try:
        h, w = img_tile.shape[:2]
        if h < TILE_SIZE or w < TILE_SIZE: return img_tile

        # 1. 텍스트 -> 비트 (9글자)
        wm_bits = text_to_bits_fixed(text, fixed_len_bytes=FIXED_BYTE_LEN)
        
        # 2. 짝수 크기 보정
        h = h - (h % 2)
        w = w - (w % 2)
        img_tile = img_tile[:h, :w]

        # 3. 워터마크 박기 (Bit Mode)
        bwm = WaterMark(password_img=PWD_IMG, password_wm=PWD_WM)
        
        unique_suffix = str(np.random.randint(0, 100000))
        temp_tile_in = os.path.join(TEMP_DIR, f"tile_in_{unique_suffix}.png")
        
        cv2.imwrite(temp_tile_in, img_tile)
        
        bwm.read_img(temp_tile_in)
        bwm.read_wm([bool(b) for b in wm_bits], mode="bit") 
        
        res_img = bwm.embed()
        
        # 4. 저장 및 정리
        res_img = np.rint(res_img).clip(0, 255).astype(np.uint8)
        
        try: os.remove(temp_tile_in)
        except: pass
            
        return res_img
    except Exception as e:
        print(f"Tile Error: {e}")
        return img_tile

def process_image_tiled(img_path, text, out_path):
    img = cv2.imread(img_path)
    if img is None: raise Exception("이미지 읽기 실패")
    
    max_dim = 1200 
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    pad_h = (TILE_SIZE - (h % TILE_SIZE)) % TILE_SIZE
    pad_w = (TILE_SIZE - (w % TILE_SIZE)) % TILE_SIZE
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    padded_h, padded_w = img_padded.shape[:2]

    for y in range(0, padded_h, TILE_SIZE):
        for x in range(0, padded_w, TILE_SIZE):
            tile = img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
            watermarked_tile = embed_tile(tile, text)
            img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE] = watermarked_tile

    final_img = img_padded[:h, :w]
    cv2.imwrite(out_path, final_img)

@app.route('/embed', methods=['POST'])
def embed():
    try:
        if 'image' not in request.files: return "No image", 400
        file = request.files['image']
        text = request.form.get('text', 'TO.000000') # 기본 예시 변경
        
        rnd = str(np.random.randint(0, 100000))
        in_path = os.path.join(TEMP_DIR, f"in_{rnd}.png")
        out_path = os.path.join(TEMP_DIR, f"out_{rnd}.png")
        
        file.save(in_path)
        process_image_tiled(in_path, text, out_path)
        
        return send_file(out_path, mimetype='image/png', as_attachment=True, download_name='secured.png')
    except Exception as e:
        return str(e), 500

@app.route('/')
def home():
    return "Watermark Server (9-Char Support)"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
