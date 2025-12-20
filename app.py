import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file
from blind_watermark import WaterMark
import cv2
import numpy as np
import requests
import shutil

app = Flask(__name__)
TEMP_DIR = "temp_server"

# 임시 폴더 초기화 (서버 시작 시 청소)
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR)

# ==========================================
# 🔐 [설정] 2단계 방어 시스템
# ==========================================
# 1. 타일링 설정 (잘라내기 방어)
TILE_SIZE = 200       # 200px 단위로 쪼개서 박음
TEXT_LEN = 8          # ID 길이
REPEAT_COUNT = 3      # 타일 내 반복 횟수

# 2. 워터마크 강도 설정 (캡처/압축 방어)
# 값이 클수록 화질은 약간 거칠어지지만, 검출률은 비약적으로 상승함
WM_STRENGTH = 3.5     
# ==========================================

def embed_tile(img_tile, text):
    """작은 타일 하나에 강력한 워터마크를 박는 함수"""
    try:
        h, w = img_tile.shape[:2]
        # 타일이 너무 작으면 패스
        if h < TILE_SIZE or w < TILE_SIZE: return img_tile

        # 텍스트 -> 비트 변환
        text = text.ljust(TEXT_LEN)[:TEXT_LEN]
        bits = []
        for char in text:
            bin_val = bin(ord(char))[2:].zfill(8)
            bits.extend([int(b) for b in bin_val])
        wm_bits = bits * REPEAT_COUNT

        # 라이브러리 객체 생성
        bwm = WaterMark(password_wm=1, password_img=1)
        
        # 임시 파일 경로 (충돌 방지용 랜덤 이름 권장되나, 단일 스레드 가정 하에 고정)
        # 실제 운영 시엔 uuid 등을 쓰는 것이 좋음
        unique_suffix = str(np.random.randint(0, 100000))
        temp_tile_in = os.path.join(TEMP_DIR, f"tile_in_{unique_suffix}.png")
        temp_tile_out = os.path.join(TEMP_DIR, f"tile_out_{unique_suffix}.png")
        
        cv2.imwrite(temp_tile_in, img_tile)
        
        bwm.read_img(temp_tile_in)
        bwm.read_wm(wm_bits, mode='bit')
        
        # 🔥 [핵심] 강도(scale)를 3.5로 설정하여 생존율 극대화
        bwm.embed(temp_tile_out, wm_content={'mode': 'bit', 'scale': WM_STRENGTH})
        
        # 결과 읽어서 리턴
        res_img = cv2.imread(temp_tile_out)
        
        # 임시 파일 정리
        try:
            os.remove(temp_tile_in)
            os.remove(temp_tile_out)
        except: pass
            
        return res_img
    except Exception as e:
        print(f"Tile Error: {e}")
        return img_tile

def process_image_tiled(img_path, text, out_path):
    """전체 이미지를 800px로 압축(리사이징) 후 타일링 처리"""
    img = cv2.imread(img_path)
    if img is None: raise Exception("이미지 파일을 읽을 수 없습니다.")
    
    # 🚀 [속도/표준화 핵심] 
    # 이미지를 강제로 800px 이하로 줄입니다.
    # 1. 무료 서버의 메모리/CPU 부담을 줄여 502 에러 방지
    # 2. 워터마크 패턴의 스케일을 일정하게 유지하여 검출률 향상
    max_dim = 800 
    
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 패딩 추가 (타일 크기의 배수로 맞춤)
    pad_h = (TILE_SIZE - (h % TILE_SIZE)) % TILE_SIZE
    pad_w = (TILE_SIZE - (w % TILE_SIZE)) % TILE_SIZE
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    padded_h, padded_w = img_padded.shape[:2]

    # 타일링 루프
    for y in range(0, padded_h, TILE_SIZE):
        for x in range(0, padded_w, TILE_SIZE):
            # 조각내기
            tile = img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # 워터마크 박기 (여기가 오래 걸림)
            watermarked_tile = embed_tile(tile, text)
            
            # 다시 붙이기
            img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE] = watermarked_tile

    # 패딩 제거 후 결과 저장
    final_img = img_padded[:h, :w]
    
    # 최종 결과물은 PNG로 저장 (서버 내부에서는 손실 없이 저장)
    # 웹으로 전송될 때는 용량이 좀 클 수 있으나, 워터마크 보존을 위해 PNG 권장
    cv2.imwrite(out_path, final_img)

@app.route('/embed', methods=['POST'])
def embed():
    """파일 업로드 방식"""
    try:
        if 'image' not in request.files: return "No image", 400
        file = request.files['image']
        text = request.form.get('text', 'User1234')
        
        # 파일명 충돌 방지
        rnd = str(np.random.randint(0, 100000))
        in_path = os.path.join(TEMP_DIR, f"in_{rnd}.png")
        out_path = os.path.join(TEMP_DIR, f"out_{rnd}.png")
        
        file.save(in_path)
        
        process_image_tiled(in_path, text, out_path)
        
        return send_file(out_path, mimetype='image/png', as_attachment=True, download_name='secured.png')
    except Exception as e:
        return str(e), 500

@app.route('/view', methods=['GET'])
def view():
    """구글 드라이브 연동 방식"""
    try:
        file_id = request.args.get('id')
        text = request.args.get('text', 'Secure')
        
        if not file_id: return "No ID", 400

        # 구글 드라이브에서 다운로드
        url = f'https://drive.google.com/uc?export=view&id={file_id}'
        resp = requests.get(url)
        if resp.status_code != 200: return "Image not found on Drive", 404
        
        # 파일명 충돌 방지
        rnd = str(np.random.randint(0, 100000))
        in_path = os.path.join(TEMP_DIR, f"in_view_{rnd}.png")
        out_path = os.path.join(TEMP_DIR, f"out_view_{rnd}.png")
        
        with open(in_path, 'wb') as f:
            f.write(resp.content)
            
        process_image_tiled(in_path, text, out_path)
        
        # 처리 후 즉시 파일 삭제 (용량 관리)
        # send_file이 파일을 읽고 나서 지우도록 하는 것은 까다로우므로
        # Render 무료 서버는 재배포 시 자동 초기화됨을 이용하거나,
        # 주기적으로 청소하는 로직이 필요함. 여기서는 유지.
        
        return send_file(out_path, mimetype='image/png')
        
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

@app.route('/')
def home():
    return "Watermark Server is Running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
