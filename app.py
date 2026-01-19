# wm_embed_server.py
# ------------------------------------------------------------
# Watermark Embedding API Server (Lightweight)
# For Google Sheets Integration
# ------------------------------------------------------------

import base64
import hashlib
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# Config & Utils
# =========================
MAX_IMAGE_PIXELS = 40_000_000

class WMConfig:
    block_px: int = 8
    tile_blocks: int = 16
    bit_pair_a: tuple = (2, 3)
    bit_pair_b: tuple = (3, 2)
    sync16: int = 0xA5C3
    ver_bits: int = 4
    id_bits: int = 20
    nonce_bits: int = 8
    crc_bits: int = 16
    msg_bits: int = 48
    coded_slots: int = 240
    code_len: int = 2 * (48 + 6) # K=7, Tail=6
    grid_period: int = 32

CFG = WMConfig()
G1 = 0o171
G2 = 0o133
K = 7
TAIL = K - 1

def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc

def int_to_bits(n: int, width: int):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        chunk = bits[i:i+8]
        for bit in chunk:
            b = (b << 1) | (bit & 1)
        if len(chunk) < 8: b <<= (8 - len(chunk))
        out.append(b)
    return bytes(out)

def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)

def rng_from_key(key: int, salt: int):
    seed = splitmix64((key & 0xFFFFFFFFFFFFFFFF) ^ (salt & 0xFFFFFFFFFFFFFFFF))
    return np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)

def parity(x: int) -> int:
    return bin(x).count("1") & 1

def conv_encode(msg_bits):
    state = 0
    out = []
    for b in msg_bits + [0]*TAIL:
        state = ((state << 1) | (b & 1)) & ((1 << K) - 1)
        out.append(parity(state & G1))
        out.append(parity(state & G2))
    return out

def sha1_of_pixels_bgr(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    payload = img_bgr.tobytes() + f"|{w}x{h}|bgr".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]

# =========================
# Core Embedding Logic
# =========================

def build_perm(key: int, cfg: WMConfig):
    rng = rng_from_key(key, 0x574D5F54494C45) # 'WM_TILE'
    return rng.permutation(cfg.tile_blocks * cfg.tile_blocks)

def build_slot_map(key: int, cfg: WMConfig):
    rng = rng_from_key(key, 0x534C4F545F4D4150) # 'SLOT_MAP'
    return rng.integers(0, cfg.code_len, size=cfg.coded_slots, dtype=np.int32)

def texture_mask(block_u8: np.ndarray) -> float:
    b = block_u8.astype(np.float32)
    v = float(np.var(b))
    return float(np.clip(v / (v + 300.0), 0.0, 1.0))

def embed_bit_in_dct(block_u8, bit, margin, a, b):
    C = cv2.dct(block_u8.astype(np.float32) - 128.0)
    ua, va = a; ub, vb = b
    ca = float(C[ua, va]); cb = float(C[ub, vb])
    
    if bit == 1:
        if ca < cb + margin:
            d = (cb + margin - ca)
            C[ua, va] += d/2; C[ub, vb] -= d/2
    else:
        if cb < ca + margin:
            d = (ca + margin - cb)
            C[ub, vb] += d/2; C[ua, va] -= d/2
            
    return np.clip(cv2.idct(C) + 128.0, 0, 255).astype(np.uint8)

def add_grid_pattern(Y_u8, key, cfg, amp):
    H, W = Y_u8.shape[:2]
    rng = rng_from_key(key, 0x475249445F504841)
    phx = float(rng.random() * 2*np.pi)
    phy = float(rng.random() * 2*np.pi)
    x = np.arange(W, dtype=np.float32)[None, :]
    y = np.arange(H, dtype=np.float32)[:, None]
    pat = np.cos(2*np.pi*x/cfg.grid_period + phx) + np.cos(2*np.pi*y/cfg.grid_period + phy)
    return np.clip(Y_u8.astype(np.float32) + (pat/2.0)*amp, 0, 255).astype(np.uint8)

def embed_process(img_bgr, wm_id, key, margin, grid_amp):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].copy()

    if grid_amp > 0:
        Y = add_grid_pattern(Y, key, CFG, amp=grid_amp)

    # Build Message
    ver = 0
    rng_nonce = rng_from_key(key, 0x4E4F4E4345)
    nonce = int(rng_nonce.integers(0, 256))
    core = int_to_bits(ver, CFG.ver_bits) + int_to_bits(wm_id, CFG.id_bits) + int_to_bits(nonce, CFG.nonce_bits)
    crc = crc16_ccitt(bits_to_bytes(core))
    msg_bits = core + int_to_bits(crc, CFG.crc_bits)
    code_bits = conv_encode(msg_bits)

    # Tiling
    perm = build_perm(key, CFG)
    slot_map = build_slot_map(key, CFG)
    sync_bits = int_to_bits(CFG.sync16, 16)
    tile_bits = np.zeros(CFG.tile_blocks * CFG.tile_blocks, dtype=np.uint8)

    for j in range(16):
        tile_bits[int(perm[j])] = sync_bits[j]
    for s in range(CFG.coded_slots):
        tile_bits[int(perm[16 + s])] = code_bits[int(slot_map[s])]

    # Embed Blocks
    H, W = Y.shape[:2]
    bxN = W // CFG.block_px
    byN = H // CFG.block_px
    Y2 = Y.copy()
    
    for by in range(byN):
        y0 = by * CFG.block_px
        for bx in range(bxN):
            x0 = bx * CFG.block_px
            bit = int(tile_bits[(by % CFG.tile_blocks) * CFG.tile_blocks + (bx % CFG.tile_blocks)])
            block = Y2[y0:y0+CFG.block_px, x0:x0+CFG.block_px]
            m = texture_mask(block)
            eff_margin = float(margin) * (0.55 + 0.90*m)
            Y2[y0:y0+CFG.block_px, x0:x0+CFG.block_px] = embed_bit_in_dct(
                block, bit, eff_margin, CFG.bit_pair_a, CFG.bit_pair_b
            )

    ycrcb[:, :, 0] = Y2
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# =========================
# API Route
# =========================

@app.route('/api/embed', methods=['POST'])
def api_embed():
    f = request.files.get("image")
    wm_id = request.form.get("id", type=int)
    key = request.form.get("key", type=int)
    margin = request.form.get("margin", type=float, default=14.0)
    grid_amp = request.form.get("grid_amp", type=float, default=1.4)

    if not f or wm_id is None or key is None:
        return jsonify({"ok": False, "reason": "Missing image, id, or key"}), 400

    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "reason": "Image decode failed"}), 400

    if img.size > MAX_IMAGE_PIXELS:
        return jsonify({"ok": False, "reason": "Image too large"}), 400

    tpl_id = sha1_of_pixels_bgr(img)

    try:
        out_bgr = embed_process(img, wm_id, key, margin, grid_amp)

        ok, buf = cv2.imencode(".png", out_bgr)
        if not ok:
            return jsonify({"ok": False, "reason": "PNG encode failed"}), 500

        b64_str = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "ok": True,
            "template_id": tpl_id,
            "wm_id": wm_id,
            "image_base64": b64_str,
            "image_mime": "image/png"
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "reason": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




