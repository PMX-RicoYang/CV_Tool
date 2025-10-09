# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from flask import Flask, request, jsonify, send_file, abort
import cv2, numpy as np, os, json, tempfile, time, math
from typing import Dict, Any, Tuple, List, Optional
import io, base64, pickle, glob
from datetime import datetime
import requests


app = Flask(__name__)

# ---- 檔案儲存設定 ----
Z0_FILE_PATH = os.path.abspath("./calibration/z0_config.json")

# ---- 預設參數（可在請求中覆蓋）----
DEFAULTS = dict(
    REAL_DIAGONAL_MM = 49.497,   # 校正片實體對角長（mm）
    REAL_DOT_LENGTH_MM = 35.0,   # 校正片圓點邊距（mm）
    FOCAL_LENGTH_PIXEL = 960.0,  # 焦距（pixel）
    SAVE_PREVIEW = False,         # 是否輸出預覽標註圖
    PREVIEW_PATH = None,         # 預覽檔路徑（None 則自動產生）
    SELECT_MODE = "first4",      # "first4" 或 "largest4"
    # Blob 偵測參數
    BLOB = dict(
        filterByColor=True, blobColor=255,
        filterByArea=True,  minArea=50, maxArea=2000,
        filterByCircularity=True, minCircularity=0.7,
        filterByInertia=False, filterByConvexity=False
    )
)  

# =========================
# 可調整參數
# =========================
CONFIG_JSON = os.path.abspath("brightness_config.json")

# PWM 韌體 API 設定（符合您提供的 GUI 調光寫法）
PWM_BASE_URL = os.environ.get("PWM_BASE_URL", "http://localhost:8765")
PWM_CHANNEL = int(os.environ.get("PWM_CHANNEL", "1"))
PWM_TIMEOUT = float(os.environ.get("PWM_TIMEOUT", "2.0"))  # 秒

# 取像預設參數
DEFAULT_CAMERA_INDEX = 137
DEFAULT_FRAME_W = 640
DEFAULT_FRAME_H = 480

SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
debug = True

# ---------- 小工具 ----------

import numpy as np
import cv2

def rect_longside_as_width(rect):
    """
    將 cv2.minAreaRect 的 rect 轉成：
      - width 一定是最長邊 (W >= H)
      - angle 落在 [0, 90)，代表「長邊相對水平線的夾角」
    並以相同的 rect 格式 ((cx, cy), (W, H), angle) 回傳。

    參數
    ----
    rect : tuple
        原始 minAreaRect 輸出：((cx, cy), (w, h), angle)，其中 angle ∈ [-90, 0)

    回傳
    ----
    tuple
        ((cx, cy), (W, H), angle_long)；W >= H 且 angle_long ∈ [0, 90)
    """
    (cx, cy), (w, h), a = rect  # a ∈ [-90, 0)

    # 若高度比寬度長，交換並把角度 +90，使「長邊」成為 width
    if h > w:
        w, h = h, w
        a = a + 90.0  # -> [0, 90)

    # 走到這裡，w 一定是長邊，因此 a 會在 [0, 90)（長邊與水平的夾角）
    # 保險起見，做個輕度正規化
    a = float(a % 180.0)
    if a >= 90.0:
        # 若意外跑到 [90,180)，等價轉回 [0,90)（仍保持長邊為 width）
        a = 180.0 - a  # 對稱映射

    return ((float(cx), float(cy)), (float(w), float(h)), float(a))

def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".z0_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # 原子替換
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _load_z0_from_file(path: str) -> Tuple[float, Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 Z0 設定檔：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "Z0" not in data:
        raise ValueError(f"Z0 設定檔缺少 'Z0' 欄位：{path}")
    return float(data["Z0"]), data

def build_blob_detector(p: Dict[str, Any]) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    for k, v in DEFAULTS["BLOB"].items():
        setattr(params, k, v)
    if p:
        for k, v in p.items():
            setattr(params, k, v)
    ver_major = int(cv2.__version__.split('.')[0])
    return cv2.SimpleBlobDetector(params) if ver_major < 3 else cv2.SimpleBlobDetector_create(params)

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(ang)               # 逆時針
    ordered = pts[order]
    idx0 = np.argmin(np.sum(ordered, axis=1))  # 左上起點
    ordered = np.concatenate([ordered[idx0:], ordered[:idx0]], axis=0)
    return ordered  # TL, TR, BR, BL

def annotate_and_save(img: np.ndarray, kps, sorted_pts: np.ndarray, text: str, out_path: str) -> str:
    out = cv2.drawKeypoints(img, kps, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ptsi = np.int32(sorted_pts.reshape(-1,1,2))
    cv2.polylines(out, [ptsi], True, (0,200,255), 2)
    cv2.line(out, tuple(ptsi[0,0]), tuple(ptsi[2,0]), (255,180,0), 2)
    cv2.line(out, tuple(ptsi[1,0]), tuple(ptsi[3,0]), (255,180,0), 2)
    y0 = 30
    for i, line in enumerate(text.split("\n")):
        cv2.putText(out, line, (10, y0 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y0 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    return out_path

def compute_height_mm_from_image(
    image_path: str,
    REAL_DIAGONAL_MM: float,
    REAL_DOT_LENGTH_MM: float,
    FOCAL_LENGTH_PIXEL: float,
    blob_cfg: Dict[str,Any],
    select_mode: str = "first4"
) -> Tuple[Dict[str,Any], int]:
    if not os.path.isfile(image_path):
        return dict(error=f"找不到圖檔：{image_path}"), 404
    img = cv2.imread(image_path)
    if img is None:
        return dict(error=f"無法讀取影像：{image_path}"), 422

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    gray_blur = cv2.GaussianBlur(gray_inv, (5,5), 1.5)

    detector = build_blob_detector(blob_cfg or {})
    keypoints = detector.detect(gray_blur)
    n_kp = len(keypoints)
    if n_kp < 4:
        return dict(error=f"圓點不足（{n_kp} < 4）", num_keypoints=n_kp), 422

    keypoints = (sorted(keypoints, key=lambda k: k.size, reverse=True)[:4]
                 if select_mode=="largest4" else keypoints[:4])

    pts = np.array([k.pt for k in keypoints], dtype=np.float32)
    sorted_pts = order_points_clockwise(pts)
    # 對角
    d1 = float(np.linalg.norm(sorted_pts[0] - sorted_pts[2]))
    d2 = float(np.linalg.norm(sorted_pts[1] - sorted_pts[3]))
    avg_diag = (d1 + d2) / 2.0
    # 四邊
    side1 = float(np.linalg.norm(sorted_pts[0] - sorted_pts[1]))
    side2 = float(np.linalg.norm(sorted_pts[1] - sorted_pts[2]))
    side3 = float(np.linalg.norm(sorted_pts[2] - sorted_pts[3]))
    side4 = float(np.linalg.norm(sorted_pts[3] - sorted_pts[0]))
    avg_side = (side1 + side2 + side3 + side4) / 4.0
    # 高度（pinhole 兩種量測取平均）
    height_mm1 = (FOCAL_LENGTH_PIXEL * REAL_DIAGONAL_MM) / max(avg_diag, 1e-6)
    height_mm2 = (FOCAL_LENGTH_PIXEL * REAL_DOT_LENGTH_MM) / max(avg_side, 1e-6)
    height_mm  = (height_mm1 + height_mm2) / 2.0

    return dict(
        ok=True,
        image_path=image_path,
        num_keypoints=n_kp,
        points_px=sorted_pts.tolist(),   # TL, TR, BR, BL
        d1_px=round(d1, 2), d2_px=round(d2, 2),
        sides_px=[round(side1,2), round(side2,2), round(side3,2), round(side4,2)],
        avg_diag_px=round(avg_diag, 2),
        avg_side_px=round(avg_side, 2),
        height_mm1=round(height_mm1, 2),
        height_mm2=round(height_mm2, 2),
        height_mm=round(height_mm, 2),
    ), 200

def measure_from_image(
    image_path: str,
    Z0: float,
    REAL_DIAGONAL_MM: float,
    REAL_DOT_LENGTH_MM: float,
    FOCAL_LENGTH_PIXEL: float,
    blob_cfg: Dict[str,Any],
    select_mode: str = "first4",
    save_preview: bool = False,
    preview_path: str | None = None
) -> Tuple[Dict[str,Any], int]:
    hinfo, code = compute_height_mm_from_image(
        image_path, REAL_DIAGONAL_MM, REAL_DOT_LENGTH_MM, FOCAL_LENGTH_PIXEL, blob_cfg, select_mode
    )
    if code != 200:
        return hinfo, code

    height_mm = float(hinfo["height_mm"])
    container_h = Z0 - height_mm
    title = f"D(avg): {hinfo['avg_diag_px']:.2f} px  Z: {height_mm:.2f} mm\nContainer Height: {container_h:.2f} mm"

    result = dict(
        ok=True,
        image_path=image_path,
        Z0_used=round(Z0, 2),
        height_mm=round(height_mm, 2),
        container_height_mm=round(container_h, 2),
        # measure_detail=hinfo  # 內含對角與邊長資訊
    )

    if save_preview:
        if not preview_path:
            base, _ = os.path.splitext(os.path.abspath(image_path))
            preview_path = base + "_preview.jpg"
        # 重新讀圖以標註
        img = cv2.imread(image_path)
        detector = build_blob_detector(blob_cfg or {})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(cv2.bitwise_not(gray), (5,5), 1.5)
        kps = detector.detect(gray_blur)
        kps = (sorted(kps, key=lambda k: k.size, reverse=True)[:4]
               if DEFAULTS["SELECT_MODE"]=="largest4" else kps[:4])
        sorted_pts = order_points_clockwise(np.array([k.pt for k in kps], dtype=np.float32))
        out_path = annotate_and_save(img, kps, sorted_pts, title, preview_path)
        result["preview_path"] = out_path

    return result, 200

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_JSON):
        return {}
    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def save_config(data: Dict[str, Any]) -> None:
    tmp_path = CONFIG_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONFIG_JSON)


# =========================
# 工具函式：亮度計算
# =========================
def _crop_roi(img: np.ndarray, roi: Optional[List[int]]) -> np.ndarray:
    if not roi:
        return img
    x, y, w, h = map(int, roi)
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    return img[y:y+h, x:x+w]


def compute_brightness(img_bgr: np.ndarray, roi: Optional[List[int]] = None, method: str = "lab_median") -> float:
    """回傳 L* 亮度（0~100）。"""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("影像為空")

    img = _crop_roi(img_bgr, roi)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)  # 0~255（OpenCV 標準）
    L_star = L * (100.0 / 255.0)

    if method == "lab_median":
        val = float(np.median(L_star))
    else:  # lab_mean（預設）
        val = float(np.mean(L_star))
    return val


# =========================
# 工具函式：取像（OpenCV）
# =========================

def capture_frame(
    device=137,                 # 例: "/dev/video137" 或 137 或 None
    frame_w: int = 640,
    frame_h: int = 480,
    warmup_frames: int = 3,
    fps: int = 30,
    prefer_mjpeg: bool = True,
) -> np.ndarray:
    """
    以 OpenCV 取一張彩色影像。
    優先用 GStreamer；若失敗且 device 是數字，才回退 V4L2。
    """
    import cv2, numpy as np

    # 構造 GStreamer 管線（MJPEG 與 YUY2 兩路嘗試）
    gst_candidates = []
    if isinstance(device, str) and device.startswith("/dev/video"):
        dev = device
    elif device is None:
        dev = "/dev/video0"
    else:
        dev = f"/dev/video{int(device)}"

    if prefer_mjpeg:
        gst_candidates.append(
            # UVC 常見 MJPEG 路線
            f"v4l2src device={dev} io-mode=2 ! "
            f"image/jpeg, width={frame_w}, height={frame_h}, framerate={fps}/1 ! "
            f"jpegparse ! jpegdec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    # 原生 RAW (YUY2) 路線
    gst_candidates.append(
        f"v4l2src device={dev} io-mode=2 ! "
        f"video/x-raw, format=YUY2, width={frame_w}, height={frame_h}, framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

    # 逐一嘗試 GStreamer 管線
    for gst_str in gst_candidates:
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try:
                for _ in range(max(0, warmup_frames)):
                    cap.read()
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.release()
                    return frame
            finally:
                cap.release()

    # 若 GStreamer 失敗，且 device 是「數字/可轉數字」，再回退到 V4L2
    try_index = None
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        try_index = int(device)

    if try_index is not None:
        cap = cv2.VideoCapture(try_index, cv2.CAP_V4L2)
        if cap.isOpened():
            # 指定 MJPG fourcc（很多 UVC 在高解析度需要）
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(max(0, warmup_frames)):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return frame

    raise RuntimeError(
        f"擷取影像失敗：無法以 GStreamer/V4L2 打開 {dev} "
        f"(w={frame_w}, h={frame_h}, fps={fps})."
    )

# def capture_frame(camera_index: int = DEFAULT_CAMERA_INDEX,
#                   frame_w: int = DEFAULT_FRAME_W,
#                   frame_h: int = DEFAULT_FRAME_H,
#                   warmup_frames: int = 3) -> np.ndarray:
#     """以 OpenCV 取一張彩色影像。必要時可改成 RealSense 管線。"""
#     # Windows 上可加 CAP_DSHOW 加速開啟：

#     gst_str = (
#         f'v4l2src device=/dev/video{camera_index} ! '
#         f'image/jpeg, width={frame_w}, height={frame_h}, framerate={5}/1 ! '
#         f'jpegdec ! '
#         f'videoconvert ! '
#         f'appsink drop=true max-buffers=1 sync=false'
#     )
#     cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

#     # cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         # 嘗試不帶 CAP_DSHOW
#         cap = cv2.VideoCapture(camera_index)
#     if not cap.isOpened():
#         raise RuntimeError(f"無法開啟攝影機 index={camera_index}")

#     try:
#         # 設定解析度（非所有裝置都保證成功）
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

#         for _ in range(max(0, warmup_frames)):
#             cap.read()

#         ok, frame = cap.read()
#         if not ok or frame is None:
#             raise RuntimeError("擷取影像失敗")
#         return frame
#     finally:
#         cap.release()


# =========================
# 工具函式：設定 PWM
# =========================
def set_pwm(value: int, channel: int = PWM_CHANNEL) -> None:
    value = int(value)
    url = f"{PWM_BASE_URL.rstrip('/')}/fw/pwm/{channel}/{value}"
    # return True
    try:
        r = requests.get(url, timeout=PWM_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"設定 PWM 失敗：{e}")    


# ---------------------------
# 超嚴格直線度 (E_line) 評估
# ---------------------------
def _chord_deviation(px: np.ndarray) -> np.ndarray:
    """全點到端點弦線的正交距離（像素）。px: [N,2]"""
    p0, p1 = px[0], px[-1]
    v = p1 - p0
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return np.zeros(len(px), dtype=np.float64)
    n = np.array([-v[1], v[0]], dtype=np.float64) / nv  # 弦線法向量（單位）
    return np.abs((px - p0) @ n)

def _mid_bend(px: np.ndarray) -> np.ndarray:
    """
    局部彎折：每個中點到『前後點之連線』的距離（像素）。
    反映局部曲率；對「一兩點彎」特別敏感。
    """
    if len(px) < 3:
        return np.zeros(1, dtype=np.float64)
    out = []
    for i in range(1, len(px)-1):
        a, b, p = px[i-1], px[i+1], px[i]
        v = b - a
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            out.append(0.0); continue
        n = np.array([-v[1], v[0]], dtype=np.float64) / nv
        out.append(abs((p - a) @ n))
    return np.array(out, dtype=np.float64)

def _block_strict_error(px: np.ndarray) -> float:
    """對單一行/列的嚴格誤差（像素）。"""
    d  = _chord_deviation(px)            # 全局彎曲
    b  = _mid_bend(px)                   # 局部彎折
    rms = float(np.sqrt(np.mean(d**2)))  # 平均彎曲
    p95 = float(np.percentile(d, 95)) if len(d) else 0.0
    mx  = float(np.max(d)) if len(d) else 0.0
    mb  = float(np.max(b)) if len(b) else 0.0
    # 嚴格聚合（對尖峰特別敏感）
    return math.sqrt(rms*rms + p95*p95 + mx*mx + 0.5*mb*mb)

def straightness_error(corners, cols, rows, mode="ultra") -> float:
    """
    E_line（像素）：對每一『列』與『行』算 _block_strict_error，
    再以嚴格方式聚合：
      - 'ultra'：取 max（最差行/列）
      - 'strict'：取 P90
      - 'mean'：平均
    """
    pts = corners.reshape(-1, 2).astype(np.float64)
    errs = []
    # 列
    for r in range(rows):
        row = np.stack([pts[r*cols + c] for c in range(cols)], axis=0)
        errs.append(_block_strict_error(row))
    # 行
    for c in range(cols):
        col = np.stack([pts[r*cols + c] for r in range(rows)], axis=0)
        errs.append(_block_strict_error(col))

    if mode == "mean":
        return float(np.mean(errs))
    if mode == "strict":
        return float(np.percentile(errs, 90))
    # default ultra
    return float(np.max(errs))

# ---------------------------
# 角點偵測與其他指標
# ---------------------------
def detect_corners(img, cols, rows) -> Tuple[bool, np.ndarray]:
    """回傳 (ok, corners[N,1,2])，角點順序預期為 row-major。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE |
            cv2.CALIB_CB_FAST_CHECK
        )
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
    return ok, corners

def mean_grid_spacing(corners, cols, rows) -> float:
    """估算平均格距(像素): 橫向與縱向鄰點距離的均值。"""
    pts = corners.reshape(-1,2)
    ds = []
    for r in range(rows):
        for c in range(cols-1):
            i = r*cols + c
            ds.append(np.linalg.norm(pts[i+1]-pts[i]))
    for c in range(cols):
        for r in range(rows-1):
            i = r*cols + c
            ds.append(np.linalg.norm(pts[i+cols]-pts[i]))
    return float(np.mean(ds)) if ds else 1.0

def homography_rms(corners, cols, rows) -> float:
    """E_homo: 以(列,行)格點座標為理想平面座標，擬合單一H到影像角點，計算RMS(像素)。"""
    img_pts = corners.reshape(-1,2)
    grid_pts = np.array([[float(c), float(r)] for r in range(rows) for c in range(cols)], dtype=np.float64)
    H, _ = cv2.findHomography(grid_pts, img_pts, method=0)
    if H is None:
        return float('inf')
    gp_h = np.hstack([grid_pts, np.ones((len(grid_pts),1))])
    proj = (gp_h @ H.T)
    proj = proj[:,:2] / proj[:,2:3]
    rms = np.sqrt(np.mean(np.sum((proj - img_pts)**2, axis=1)))
    return float(rms)

def quick_calibrate_dist_shift(img, corners, cols, rows, square_size) -> float:
    """
    E_dist: 快速單影像校正 → 「含畸變 vs 零畸變」平均像素差。
    """
    h, w = img.shape[:2]
    objp = np.array([[c*square_size, r*square_size, 0.0]
                     for r in range(rows) for c in range(cols)], dtype=np.float32)
    imgp = corners.reshape(-1,1,2).astype(np.float32)
    objpoints = [objp]; imgpoints = [imgp]
    K = np.array([[w, 0, w/2],
                  [0, w, h/2],
                  [0, 0,   1 ]], dtype=np.float64)
    dist = np.zeros((5,1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [imgp], (w,h), K, dist, flags=flags, criteria=criteria
    )
    if not np.isfinite(ret):
        return float('nan')
    rvec, tvec = rvecs[0], tvecs[0]
    pts_d, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    zero = np.zeros_like(dist)
    pts_0, _ = cv2.projectPoints(objp, rvec, tvec, K, zero)
    delta = np.linalg.norm(pts_d.reshape(-1,2) - pts_0.reshape(-1,2), axis=1)
    return float(np.mean(delta))

def compute_score(E_line_px, E_homo_px, E_dist_px, spacing,
                  mode="strict", T=None, k=None, weights=(0.7, 0.1, 0.2)) -> Tuple[int, Dict[str, Any]]:
    """
    嚴格版：先正規化(除以平均格距) → 加權L2彙總 → logistic 映射成 0~100。
      mode: "default"|"strict"|"ultra" 決定 T,k 預設
      T: 50分門檻 (越小越嚴格)；k: 斜率 (越大越敏感)
    建議:
      default: T=0.060, k=60
      strict : T=0.040, k=90
      ultra  : T=0.030, k=120
    """
    denom = max(spacing, 1e-6)
    E_line = E_line_px / denom
    E_homo = E_homo_px / denom
    E_dist = (E_dist_px / denom) if np.isfinite(E_dist_px) else E_line
    w1, w2, w3 = weights
    E = math.sqrt((w1*E_line)**2 + (w2*E_homo)**2 + (w3*E_dist)**2)

    if T is None or k is None:
        if mode == "ultra":
            T, k = 0.030, 120
        elif mode == "strict":
            T, k = 0.040, 90
        else:
            T, k = 0.060, 60

    score = int(round(100.0 / (1.0 + math.exp(k * (E - T)))))
    score = max(0, min(100, score))
    return score, {
        "E_line": E_line, "E_homo": E_homo, "E_dist": E_dist,
        "E_total": E, "midpoint_T": T, "slope_k": k, "weights": [w1,w2,w3]
    }

def visualize(img, corners, cols, rows, out_path) -> None:
    """畫出列/行擬合與角點（肉眼檢視彎曲）。"""
    vis = img.copy()
    pts = corners.reshape(-1,2)
    # 角點
    for (x,y) in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0,255,255), -1)
    # 列擬合
    for r in range(rows):
        row = np.array([pts[r*cols + c] for c in range(cols)], dtype=np.float64)
        mu = row.mean(axis=0)
        Q = row - mu
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        d = vh[0]
        p1 = (mu - 2000*d).astype(int)
        p2 = (mu + 2000*d).astype(int)
        cv2.line(vis, tuple(p1), tuple(p2), (0,255,0), 2)
    # 行擬合
    for c in range(cols):
        col = np.array([pts[r*cols + c] for r in range(rows)], dtype=np.float64)
        mu = col.mean(axis=0)
        Q = col - mu
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        d = vh[0]
        p1 = (mu - 2000*d).astype(int)
        p2 = (mu + 2000*d).astype(int)
        cv2.line(vis, tuple(p1), tuple(p2), (255,0,0), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

# ---------------------------
# 核心計算流程（給 API 用）
# ---------------------------
def compute_report(params: Dict[str, Any]) -> Dict[str, Any]:
    image_path = params.get("image_path")
    cols = int(params.get("cols"))
    rows = int(params.get("rows"))
    square_size = float(params.get("square_size", 1.0))
    out_dir = params.get("out_dir", "distortion_results")
    eline_mode = params.get("eline_mode", "ultra")
    score_mode = params.get("score_mode", "strict")
    score_mid = params.get("score_midpoint", 0.05)
    score_slope = params.get("score_slope", 80)

    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Unable to read the image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"OpenCV cannot load the image: {image_path}")

    ok, corners = detect_corners(img, cols, rows)
    if not ok:
        raise RuntimeError("Corner detection failed; please check whether cols/rows are set correctly, and whether the image quality, exposure, and chessboard clarity are sufficient.")

    spacing = mean_grid_spacing(corners, cols, rows)
    E_line_px = straightness_error(corners, cols, rows, mode=eline_mode)
    E_homo_px = homography_rms(corners, cols, rows)
    E_dist_px = quick_calibrate_dist_shift(img, corners, cols, rows, square_size)

    # base = os.path.splitext(os.path.basename(image_path))[0]
    # vis_path = os.path.join(out_dir, f"{base}_visualization.png")
    # os.makedirs(out_dir, exist_ok=True)
    # visualize(img, corners, cols, rows, vis_path)

    score, parts = compute_score(E_line_px, E_homo_px, E_dist_px, spacing,
                                 mode=score_mode, T=score_mid, k=score_slope,
                                 weights=(0.7, 0.1, 0.2))  # E_line 權重高

    report = {
        "ok": True,
        "score_0to100": score,
        # "image": os.path.abspath(image_path),
        # "cols": cols, "rows": rows, "square_size": square_size,
        # "mean_spacing_px": spacing,
        # "E_line_px": E_line_px, "E_homo_px": E_homo_px, "E_dist_px": E_dist_px,
        # "E_line_norm": parts["E_line"], "E_homo_norm": parts["E_homo"], "E_dist_norm": parts["E_dist"],
        # "E_total_norm": parts["E_total"],
        # "score_mode": score_mode,
        # "score_midpoint_T": parts["midpoint_T"],
        # "score_slope_k": parts["slope_k"],
        # "weights": parts["weights"],
        # "visualization_png": os.path.abspath(vis_path),
        # "notes": "分數越高=畸變越小；建議多張/多視角取中位數更穩。"
    }
    return report        


# ===============================
# 通用工具
# ===============================

def norm_dir(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))

def joinp(d: str, f: str) -> str:
    return os.path.normpath(os.path.join(d, f))

def ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def imread_safe(path: str):
    """讀 BGR（不保留 alpha）；適合一般照片/背景圖（含路徑含中文的情況）。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imread_safe_any(path: str):
    """讀取影像，保留 alpha（若有）。用於讀 DB 物件圖。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

def split_bgra(img_any: np.ndarray):
    """輸入任意通道影像 → (BGR, alpha或None)。灰階會轉 BGR。"""
    if img_any is None:
        return None, None
    if img_any.ndim == 3 and img_any.shape[2] == 4:
        return img_any[:, :, :3], img_any[:, :, 3]
    elif img_any.ndim == 3 and img_any.shape[2] == 3:
        return img_any, None
    elif img_any.ndim == 2:
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR), None
    else:
        return None, None

# ===============================
# 幾何/顏色/特徵（均可受 mask 約束）
# ===============================

def contour_from_mask(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def minrect_metrics(cnt):
    if cnt is None:
        return None, 0.0, float("nan")
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    area = float(cv2.contourArea(cnt))
    if w <= 0 or h <= 0:
        ar = float("nan")
    else:
        ar = float(max(w, h) / max(1e-6, min(w, h)))
    return rect, area, ar

def match_shape_distance(cntA, cntB):
    try:
        return float(cv2.matchShapes(cntA, cntB, cv2.CONTOURS_MATCH_I3, 0.0))
    except cv2.error:
        return float("inf")


# ---- Hue 直方圖：保留「無彩度 bin」而非移除白/灰/黑 ----
# def compute_nonwhite_hue_sig(img_bgr, obj_mask=None, h_bins=18, s_min=40, v_min=30, min_count=10,
#                              v_black=20):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     H, S, V = cv2.split(hsv)

#     obj = (obj_mask > 0) if obj_mask is not None else np.ones_like(H, bool)
#     denom = max(1, int(obj.sum()))

#     # 將無彩度拆成「黑」與「灰/白」
#     black = (V < v_black) & obj
#     graywhite = (S < s_min) & (~black) & obj
#     achrom = black | graywhite

#     hue_sel = (~achrom) & obj
#     cnt_colored = int(hue_sel.sum())
#     mask_u8 = (hue_sel.astype(np.uint8) * 255)

#     hist = cv2.calcHist([H], [0], mask_u8, [h_bins], [0, 180]).astype(np.float32).flatten()

#     # 把灰白與黑分開當作兩個末端 bin
#     graywhite_count = float(graywhite.sum())
#     black_count = float(black.sum())
#     vec = np.concatenate([hist, [graywhite_count, black_count]], axis=0)  # 長度 = h_bins + 2

#     if int(vec.sum()) < min_count:
#         colored_frac = float(cnt_colored) / float(denom)
#         black_frac = float(black_count) / float(denom)
#         return None, colored_frac

#     sig = vec / (vec.sum() + 1e-6)
#     colored_frac = float(cnt_colored) / float(denom)
#     black_frac = float(black_count) / float(denom)
#     return sig.tolist(), colored_frac        
def compute_nonwhite_hue_sig(
    img_bgr,
    obj_mask=None,
    h_bins=9,  #18  24
    s_min=40,     # 舊參數保留：映射為 C* 門檻（見下）
    v_min=30,     # 兼容參數，未使用
    min_count=10,
    v_black=20    # 舊參數保留：映射為 L* 門檻（見下）
):
    """
    以 CIE L*a*b* → LCh 的 hue 建直方圖，並用 L* 判黑、C* 判灰白。
    比 HSV 更符合人眼感知，但不需背景圖。
    
    參數:
      img_bgr   : BGR 影像 (uint8)
      obj_mask  : 0/255 的 uint8 mask（無則對整張圖統計）
      h_bins    : 色相分箱數（0~360°）
      s_min     : 舊 HSV「S」門檻，這裡映射為 C*（彩度）門檻用
      v_min     : 兼容保留，未使用
      min_count : 最少有效像素（避免過少導致不穩）
      v_black   : 舊 HSV「V」門檻，這裡映射為 L*（明度）門檻用

    回傳:
      sig           : 長度 = h_bins + 2 的機率向量（最後兩格：灰白、黑）
      colored_frac  : 有彩度像素 / mask 內像素 的比例
    """
    # --- 1) BGR → Lab（OpenCV 的 L:0~255；a,b:0~255）---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)   # 正規化為 L* ∈ [0,100]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    # --- 2) Lab → LCh（C* 彩度、h 色相角）---
    C = np.sqrt(a*a + b*b)
    h = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0  # 0~360°

    # --- 3) 物件區域 ---
    if obj_mask is not None:
        obj = (obj_mask > 0)
    else:
        obj = np.ones(L.shape, dtype=bool)
    denom = int(obj.sum())
    if denom <= 0:
        return None, 0.0

    # --- 4) 門檻映射（讓舊參數可用在 L*, C* 世界）---
    # 黑色: 以 L* 判斷；v_black=20 → L*≈7.84（可依實務調整）
    L_black = float(v_black) * (100.0 / 255.0)
    # 灰白: 以 C* 判斷；把 s_min 約略映成 C*，並設下限避免過嚴
    # 建議: s_min=40 → C_min≈9；你可依場景微調係數 0.22 與下限 6
    C_min = max(6.0, 0.22 * float(s_min))

    # --- 5) 分群 ---
    black = (L < L_black) & obj
    graywhite = (C < C_min) & (~black) & obj
    colored = (~black) & (~graywhite) & obj

    cnt_colored = int(colored.sum())
    cnt_graywhite = int(graywhite.sum())
    cnt_black = int(black.sum())

    total = cnt_colored + cnt_graywhite + cnt_black
    if total < min_count:
        colored_frac = cnt_colored / max(1.0, float(denom))
        return None, colored_frac

    # --- 6) 對「有彩像素」建 hue 直方圖 ---
    h32 = h.astype(np.float32)
    mask_u8 = (colored.astype(np.uint8) * 255)
    hist = cv2.calcHist([h32], [0], mask_u8, [int(h_bins)], [0, 360]).astype(np.float32).flatten()

    # --- 7) 拼成特徵向量：[hue_hist, graywhite_count, black_count] ---
    vec = np.concatenate([hist, [float(cnt_graywhite), float(cnt_black)]], axis=0)

    # --- 8) 正規化為機率向量 ---
    sig = vec / (vec.sum() + 1e-6)
    colored_frac = cnt_colored / float(denom)

    return sig.tolist(), colored_frac

def hue_distance(sig1, sig2):
    a = np.asarray(sig1, np.float32).reshape(-1, 1)
    b = np.asarray(sig2, np.float32).reshape(-1, 1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

# ---- 特徵描述子（預設改用 ORB，速度較穩；若要 AKAZE 可自行切換） ----

def akaze_desc(img_bgr, mask=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # akaze = cv2.AKAZE_create()
    akaze = cv2.ORB_create(nfeatures=500)
    kp, des = akaze.detectAndCompute(gray, mask)
    return kp, des

def feat_good_matches(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good += 1
    return good

# ===============================
# 從圖片萃取主輪廓（支援 alpha）
# ===============================

def extract_main_contour_from_image(img_any):
    bgr, alpha = split_bgra(img_any)
    if bgr is None:
        return None

    if alpha is not None:
        mask = (alpha > 0).astype(np.uint8) * 255
        return contour_from_mask(mask)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for th in (th1, th2):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th[:] = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    c1 = contour_from_mask(th1)
    c2 = contour_from_mask(th2)
    if c1 is None and c2 is None:
        return None
    if c1 is None:
        return c2
    if c2 is None:
        return c1
    return c1 if cv2.contourArea(c1) >= cv2.contourArea(c2) else c2

# ===============================
# DB 建立（形狀/大小/顏色/特徵）— 將「mask」內資訊存成描述子
# ===============================

def build_db_descriptors(db_dir):
    db_dir = norm_dir(db_dir)
    if not os.path.isdir(db_dir):
        ensure_dir(db_dir)
        return []

    descs = []
    for fn in os.listdir(db_dir):
        if os.path.splitext(fn)[1].lower() not in SUPPORTED:
            continue
        p = joinp(db_dir, fn)
        if not os.path.isfile(p):
            continue

        img_any = imread_safe_any(p)
        if img_any is None:
            continue
        img_bgr, alpha = split_bgra(img_any)
        if img_bgr is None:
            continue

        if alpha is not None:
            obj_mask = (alpha > 0).astype(np.uint8) * 255
            cnt = contour_from_mask(obj_mask)
        else:
            cnt = extract_main_contour_from_image(img_any)
            obj_mask = None
            if cnt is not None:
                obj_mask = np.zeros(img_bgr.shape[:2], np.uint8)
                cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        if cnt is None:
            continue

        rect, area, ar = minrect_metrics(cnt)
        hsig, cfrac = compute_nonwhite_hue_sig(img_bgr, obj_mask=obj_mask)
        kp, des = akaze_desc(img_bgr, mask=obj_mask)

        descs.append({
            "file": fn,
            "contour": cnt,
            "area": area,
            "ar": ar,
            "rect": rect,
            "h_sig": hsig,
            "colored_frac": cfrac,
            "kp": kp,
            "des": des,
        })
    return descs

# ===============================
# ΔE (Lab) 背景差異判定
# ===============================

def deltaE76_lab(img_bgr, bg_bgr):
    lab1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab1 - lab2
    de = np.sqrt(np.sum(diff * diff, axis=-1))
    return de

def deltaE2000_lab(img_bgr, bg_bgr):
    """
    兩張 BGR 圖轉 Lab，計算像素級 CIEDE2000 (ΔE00)。
    回傳：HxW float32
    參考：Sharma, Wu, Dalal (2005)
    """
    # OpenCV 的 Lab：L ∈ [0,100] 映到 [0,255]；a,b 平移縮放到 [0,255] → 先轉回標準量綱
    lab1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab2 = cv2.cvtColor(bg_bgr,  cv2.COLOR_BGR2LAB).astype(np.float64)

    # 轉成 CIE Lab 真實範圍
    L1 = lab1[..., 0] * (100.0 / 255.0)
    a1 = lab1[..., 1] - 128.0
    b1 = lab1[..., 2] - 128.0

    L2 = lab2[..., 0] * (100.0 / 255.0)
    a2 = lab2[..., 1] - 128.0
    b2 = lab2[..., 2] - 128.0

    # 步驟 1：預備量
    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    Cm = 0.5 * (C1 + C2)

    G = 0.5 * (1.0 - np.sqrt((Cm**7) / (Cm**7 + 25.0**7 + 1e-15)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p*a1p + b1*b1)
    C2p = np.sqrt(a2p*a2p + b2*b2)

    # 色相角（度）
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # 步驟 2：差值
    dLp = L2 - L1
    dCp = C2p - C1p

    # Δhp（注意跨 360/0 的情況）
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, dh)
    dh = np.where(dh < -180.0, dh + 360.0, dh)
    dHp = 2.0 * np.sqrt(C1p*C2p) * np.sin(np.radians(dh) / 2.0)

    # 平均 L', C', h'
    Lpm = 0.5 * (L1 + L2)
    Cpm = 0.5 * (C1p + C2p)

    # h' 平均（同樣注意跨界）
    hsum = h1p + h2p
    hp_diff = np.abs(h1p - h2p)
    hpm = np.where((C1p*C2p) == 0.0, hsum, np.where(hp_diff <= 180.0, 0.5*hsum,
           np.where(hsum < 360.0, 0.5*(hsum + 360.0), 0.5*(hsum - 360.0))))

    # 步驟 3：加權與旋轉項
    T = (1
         - 0.17*np.cos(np.radians(hpm - 30))
         + 0.24*np.cos(np.radians(2*hpm))
         + 0.32*np.cos(np.radians(3*hpm + 6))
         - 0.20*np.cos(np.radians(4*hpm - 63)))

    Sl = 1 + (0.015 * (Lpm - 50)**2) / np.sqrt(20 + (Lpm - 50)**2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T

    del_th = 30.0 * np.exp(-((hpm - 275.0)/25.0)**2)
    Rc = 2.0 * np.sqrt((Cpm**7) / (Cpm**7 + 25.0**7 + 1e-15))
    Rt = -Rc * np.sin(2.0 * np.radians(del_th))

    # kL, kC, kH 取 1（標準觀察條件）
    kL = kC = kH = 1.0

    # 步驟 4：ΔE00
    dE = np.sqrt(
        (dLp/(kL*Sl))**2 +
        (dCp/(kC*Sc))**2 +
        (dHp/(kH*Sh))**2 +
        Rt * (dCp/(kC*Sc)) * (dHp/(kH*Sh))
    )

    return dE.astype(np.float32)

# ===============================
# 旋轉框裁圖（正立）
# ===============================

def crop_rotated_rect(img, rect, pad=4):
    (cx, cy), (w, h), ang = rect
    w0, h0 = int(round(w)), int(round(h))
    angle = ang
    if w0 < h0:
        angle = ang + 90.0
        w0, h0 = h0, w0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    patch = cv2.getRectSubPix(rotated, (max(1, int(round(w0 + 2 * pad))), max(1, int(round(h0 + 2 * pad)))), (cx, cy))
    return patch

# ===============================
# 重疊去重（旋轉框 IoU）
# ===============================

def _rect_dict_area(rdict):
    rc = rdict["rect"]
    return float(rc["w"]) * float(rc["h"])

def _rect_dict_to_cvtuple(rc):
    return ((float(rc["cx"]), float(rc["cy"])), (float(rc["w"]), float(rc["h"])), float(rc["angle"]))

def _rotated_iou_rectdict(a, b):
    ra = _rect_dict_to_cvtuple(a["rect"])
    rb = _rect_dict_to_cvtuple(b["rect"])
    pa = cv2.boxPoints(ra).astype(np.float32)
    pb = cv2.boxPoints(rb).astype(np.float32)
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter_area = 0.0 if inter_area is None else float(inter_area)
    Aa = cv2.contourArea(pa); Ab = cv2.contourArea(pb)
    if Aa <= 0 or Ab <= 0:
        return 0.0
    union = Aa + Ab - inter_area
    return 0.0 if union <= 0 else inter_area / union

def dedup_overlapping_rects(rects, iou_thr=0.98):
    n = len(rects)
    if n <= 1:
        return list(rects), []

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _rotated_iou_rectdict(rects[i], rects[j]) >= iou_thr:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    keep_mask = [True] * n
    pruned = []

    def area_of(idx):
        return _rect_dict_area(rects[idx])

    def worse_idx_by_score(a, b):
        sa = rects[a].get("score")
        sb = rects[b].get("score")
        if sa is not None and sb is not None and sa != sb:
            return a if sa > sb else b
        return max(a, b)

    for s in range(n):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        if len(comp) == 1:
            continue

        deg = {u: len(adj[u]) for u in comp}

        if len(comp) == 2 and all(deg[u] == 1 for u in comp):
            i, j = comp
            ai, aj = area_of(i), area_of(j)
            if ai > aj:
                drop = j
            elif aj > ai:
                drop = i
            else:
                drop = worse_idx_by_score(i, j)
            if keep_mask[drop]:
                keep_mask[drop] = False
                pruned.append(rects[drop])
        else:
            areas = [(area_of(u), u) for u in comp]
            max_area = max(a for a, _ in areas)
            cands = [u for a, u in areas if a == max_area]
            if len(cands) == 1:
                drop = cands[0]
            else:
                drop = cands[0]
                for u in cands[1:]:
                    drop = worse_idx_by_score(drop, u)
            if keep_mask[drop]:
                keep_mask[drop] = False
                pruned.append(rects[drop])

    kept = [r for idx, r in enumerate(rects) if keep_mask[idx]]
    return kept, pruned

# ===============================
# 註冊流程（只存/只比對 mask 範圍）
# ===============================

def register_object(cand_img, cand_cnt, cand_mask, cand_rect, db_descs,
                    shape_thr, ar_tol, area_tol, color_thr, min_colored_frac,
                    feat_ratio, feat_min_matches, db_dir, save_ext=".png"):
    rect, area_mask, ar_mask = minrect_metrics(cand_cnt)

    hsig, cfrac = compute_nonwhite_hue_sig(cand_img, obj_mask=cand_mask)
    kp, des = akaze_desc(cand_img, mask=cand_mask)

    if debug:
        print("[REG] ================= CANDIDATE =================")
        print(f"[REG] area={area_mask:.1f}, ar={ar_mask:.4f}, colored_frac={cfrac:.4f}, kp={len(kp) if kp else 0}")
        print(f"[REG] thr: shape<={shape_thr:.3f}, ar_tol<={ar_tol:.3f}, area_tol<={area_tol:.3f}, hue<={color_thr:.3f}, "
              f"min_col_frac>={min_colored_frac:.3f}, feat_ratio={feat_ratio:.2f}, feat_min={feat_min_matches}")

    best = None
    existed = False

    for d in db_descs:
        d_shape = match_shape_distance(cand_cnt, d["contour"])
        d_ar    = abs(ar_mask   - d["ar"])   / max(d["ar"],   1e-6)
        d_area  = abs(area_mask - d["area"]) / max(d["area"], 1e-6)

        d_hue = None
        color_ok = True
        color_eval = False
        if (hsig is not None) and (d["h_sig"] is not None) and (cfrac >= min_colored_frac) and (d["colored_frac"] >= min_colored_frac):
            d_hue = hue_distance(hsig, d["h_sig"])
            color_ok = (d_hue <= color_thr)
            color_eval = True

        # good = feat_good_matches(des, d["des"], ratio=feat_ratio)
        # feat_ok = (good >= feat_min_matches)

        shape_ok = (d_shape <= shape_thr)
        ar_ok    = (d_ar    <= ar_tol)
        area_ok  = (d_area  <= area_tol)
        pass_all = shape_ok and ar_ok and area_ok and color_ok #and feat_ok

        score = d_shape + 0.5*d_ar + 0.25*d_area + (0.5*(d_hue if d_hue is not None else 0.0)) #+ (0.02 * max(0, feat_min_matches - good))

        if debug:
            print(f"[REG] db={d['file']}")
            print(f"[REG]   shape={d_shape:.3f} <= {shape_thr:.3f} ? {shape_ok}")
            print(f"[REG]   ar_diff={d_ar:.3f} <= {ar_tol:.3f} ? {ar_ok}")
            print(f"[REG]   area_diff={d_area:.3f} <= {area_tol:.3f} ? {area_ok}")
            if color_eval:
                print(f"[REG]   hue={d_hue:.3f} <= {color_thr:.3f} ? {color_ok}  (cfrac cand/db={cfrac:.3f}/{d['colored_frac']:.3f})")
            else:
                print(f"[REG]   hue=SKIP  (cfrac cand/db={cfrac:.3f}/{d['colored_frac']:.3f}, min={min_colored_frac:.3f})")
            # print(f"[REG]   feat_good={good} >= {feat_min_matches} ? {feat_ok}  (ratio={feat_ratio:.2f})")
            print(f"[REG] -----------------  PASS_ALL={pass_all}  score={score:.3f}")

        info = {
            "same_db_file": d["file"],
            "score": float(score),
            # "shape": float(d_shape),
            # "ar": float(d_ar),
            # "area": float(d_area),
            # "hue": (None if d_hue is None else float(d_hue)),
            # "feat_good": int(good),
            "pass":{
                "shape_ok": bool(shape_ok),
                "ar_ok": bool(ar_ok),
                "area_ok": bool(area_ok),
                "color_ok": bool(color_ok),
                # "feat_ok": bool(feat_ok),
                "pass_all": bool(pass_all), 
            }
        }
        if (best is None) or (info["score"] < best["score"]):
            best = info
        if pass_all:
            existed = True
            if debug:
                print(f"[REG] ==> EXIST: matched {d['file']} (停止搜尋)")
            break

    reg_out = {"action": "skip", "detail": None}
    if existed:
        reg_out = {"action": "exist", "detail": best}
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"obj_{ts}"
        k = 0
        while True:
            name = f"{base}_{k}{save_ext}" if k > 0 else f"{base}{save_ext}"
            outp = joinp(db_dir, name)
            if not os.path.exists(outp):
                break
            k += 1
        ensure_dir(db_dir)

        if save_ext.lower() == ".png":
            alpha = (cand_mask > 0).astype(np.uint8) * 255
            bgr = cand_img.copy()
            bgr[alpha == 0] = 0
            bgra = np.dstack([bgr, alpha])
            cv2.imwrite(outp, bgra)
        else:
            bgr = cand_img.copy()
            bgr[(cand_mask == 0)] = 0
            cv2.imwrite(outp, bgr)

        reg_out = {"action": "saved", "saved_file": os.path.basename(outp), "detail": best}
        if debug:
            print(f"[REG] ==> NEW: saved as {os.path.basename(outp)}; best_candidate={best}")
    return reg_out

# ===============================
# 載入 masks
# ===============================

def load_masks_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    masks = data.get("masks")
    H = int(data.get("height",480))
    W = int(data.get("width", 640))
    n = int(data.get("amount", 100))
    # H = int(data.get("height"))    #RICO
    # W = int(data.get("width"))
    # n = int(data.get("amount"))   
    if isinstance(masks, np.ndarray):
        arr = masks
        if arr.ndim == 2 and n == 1:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"未知的 masks 形狀：{arr.shape}")
        if arr.shape[0] != n or arr.shape[-2:] != (H, W):
            if arr.shape[:2] == (H, W) and arr.shape[2] == n:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                raise ValueError(f"masks 尺寸與 height/width/amount 不一致：{arr.shape} vs {(n, H, W)}")
        arr = (arr > 0).astype(np.uint8) * 255
        return arr
    elif isinstance(masks, list):
        out = []
        for m in masks:
            a = np.asarray(m)
            if a.shape != (H, W):
                a = cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            out.append(((a > 0).astype(np.uint8) * 255))
        arr = np.stack(out, axis=0).astype(np.uint8)
        return arr
    else:
        raise ValueError(f"masks 型態不支援：{type(masks)}")


def load_masks_from_request(payload: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """回傳 (n,H,W) 的 uint8 0/255。支援 pkl_path / mask_paths / masks 內嵌。"""
    if "mask_pkl_path" in payload and payload["mask_pkl_path"]:
        return load_masks_pkl(payload["mask_pkl_path"])  # 會自帶尺寸，呼叫端再視需要 resize

    if "mask_paths" in payload and payload["mask_paths"]:
        arrs = []
        for p in payload["mask_paths"]:
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(f"讀不到 mask：{p}")
            if m.ndim == 3:
                # 若為彩色/含 alpha，取灰或 alpha
                if m.shape[2] == 4:
                    m = m[:, :, 3]
                else:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            arrs.append(((m > 0).astype(np.uint8) * 255))
        if not arrs:
            raise ValueError("mask_paths 為空")
        return np.stack(arrs, axis=0).astype(np.uint8)

    # 內嵌（與 pkl 結構相似）
    if "masks" in payload and payload["masks"]:
        mobj = payload["masks"]
        masks = mobj.get("masks")
        if masks is None:
            raise ValueError("masks 物件缺少 'masks' 欄位")
        arr = np.asarray(masks)
        if arr.ndim == 2:
            arr = arr[None, ...]
        arr = (arr > 0).astype(np.uint8) * 255
        # 尺寸對齊
        out = []
        for i in range(arr.shape[0]):
            a = arr[i]
            if a.shape != (H, W):
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append(a)
        return np.stack(out, axis=0).astype(np.uint8)

    raise ValueError("未提供有效的 mask 資訊（mask_pkl_path / mask_paths / masks 三擇一）")


def is_irregular_mask(cnt, area, solidity_thr: float = 0.85) -> bool:
    """
     夠紮實(solidity) ⇒ 視為規則；否則不規則。
    參數：
      - solidity_thr: 用於「紮實度」門檻（建議 0.80~0.90）
    回傳：
      - True  = 不規則（應剔除）
      - False = 規則（保留）
    """
    # 紮實度（過濾破碎/鋸齒）
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = area / hull_area
    if solidity < solidity_thr:   # 可在 0.80~0.90 間調
        return True
        
    is_regular = solidity >= solidity_thr
    return (not is_regular)

# def is_irregular_mask(mask_u8, area_thr=0.6):
#     """
#     判斷 mask 是否過於不規則。
#     方法：取最大輪廓 → 與其近似多邊形或最小外接矩形比較面積。
#     如果填滿比例太低，視為不規則。
#     """
#     cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return True  # 空的就當不合格

#     cnt = max(cnts, key=cv2.contourArea)
#     area = cv2.contourArea(cnt)

#     # 最小外接矩形
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     rect_area = cv2.contourArea(box)

#     if rect_area <= 1e-6:
#         return True

#     fill_ratio = area / rect_area  # 填滿程度
#     return fill_ratio < area_thr
# ===============================
# 主流程（單組）
# ===============================

def process_single(image_path, bg_path, db_dir,
                   masks_arr: np.ndarray,
                   shape_thr=0.50, ar_tol=0.15, area_tol=0.50,
                   color_thr=0.40, min_colored_frac=0.02,
                   min_mask_area=500, max_mask_area=102400, solidity_thr=0.9, roi=None,
                   de_thr=16.0, de_frac=0.85, register=False,
                   feat_ratio=0.75, feat_min_matches=5, save_ext=".png",
                   out_path: Optional[str] = None,
                   return_image: bool = False):

    print("process_single")               
    t_all0 = time.time()

    t0 = time.time()
    img = imread_safe(image_path)
    if img is None:
        raise FileNotFoundError(f"讀不到測試圖：{image_path}")
    H, W = img.shape[:2]
    t_img = (time.time() - t0) * 1000

    t0 = time.time()
    bg = imread_safe(bg_path)
    if bg is None:
        raise FileNotFoundError(f"讀不到背景圖：{bg_path}")
    if bg.shape[:2] != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    t_bg = (time.time() - t0) * 1000

    # ROI
    roi_mask = None
    roi_clamped = None
    if roi is not None:
        x, y, w_, h_ = roi
        x = max(0, min(W - 1, int(x))); y = max(0, min(H - 1, int(y)))
        w_ = max(0, min(W - x, int(w_))); h_ = max(0, min(H - y, int(h_)))
        if w_ > 0 and h_ > 0:
            roi_clamped = (x, y, w_, h_)
            roi_mask = np.zeros((H, W), np.uint8)
            cv2.rectangle(roi_mask, (x, y), (x + w_ - 1, y + h_ - 1), 255, thickness=-1)

    # 調整 mask 尺寸
    if masks_arr.shape[-2:] != (H, W):
        masks_arr = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in masks_arr], axis=0)


    drawn = img.copy()
    rects_out = []
    all_masks_out = []
    support_data = {}

    # ΔE
    t0 = time.time()
    # de_map = deltaE76_lab(img, bg)
    de_map = deltaE2000_lab(img, bg)
    t_de = (time.time() - t0) * 1000

    t_loop = 0.0
    for i, m in enumerate(masks_arr):
        t_l0 = time.time()
        if roi_mask is not None and not np.any((m > 0) & (roi_mask == 255)):
        # if roi_mask is not None and np.any((m > 0) & (roi_mask == 0)):
            t_loop += (time.time() - t_l0) * 1000
            continue

        # # 新增：判斷 mask 是否不規則
        # if is_irregular_mask(m, irregular_thr):   # 0.6 可調，越低越寬鬆
        #     continue

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mm = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            t_loop += (time.time() - t_l0) * 1000
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area_mask = float(cv2.contourArea(cnt))
        if area_mask < min_mask_area or area_mask > max_mask_area:
            t_loop += (time.time() - t_l0) * 1000
            continue

        # 判斷 mask 是否不規則
        if is_irregular_mask(cnt, area_mask, solidity_thr):
            continue

        rect = cv2.minAreaRect(cnt)
        # rect = rect_longside_as_width(rect)   #rico 轉換
        box = cv2.boxPoints(rect).astype(np.int32)
        (rw, rh) = rect[1]
        ar_mask = float(max(rw, rh) / max(1e-6, min(rw, rh)))

        cv2.drawContours(drawn, [box], 0, (255, 0, 0), 2)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.putText(drawn, f"#{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        all_masks_out.append({
            "mask_index": int(i),
            "rect": {"cx": float(rect[0][0]), "cy": float(rect[0][1]), "w": float(rw), "h": float(rh), "angle": float(rect[2])},
            "area": float(area_mask),
            "ar": float(ar_mask)
        })

        mask_bool = (mm > 0)
        pix = int(np.count_nonzero(mask_bool))
        if pix == 0:
            t_loop += (time.time() - t_l0) * 1000
            continue
        de_vals = de_map[mask_bool]
        frac = float(np.count_nonzero(de_vals >= de_thr)) / float(pix)
        de_mean = float(np.median(de_vals)) if de_vals.size else 0.0
        
        is_object = (frac >= de_frac)

        if not is_object:
            t_loop += (time.time() - t_l0) * 1000
            continue

        cv2.drawContours(drawn, [box], 0, (0, 200, 0), 3)

        support_data[i] = {
            "rect": rect,
            "cnt": cnt,
            "mask": mm.copy(),
        }

        rects_out.append({
            "mask_index": int(i),
            "rect": {"cx": float(rect[0][0]), "cy": float(rect[0][1]), "w": float(rw), "h": float(rh), "angle": float(rect[2])},
            # "area": float(area_mask),
            # "ar": float(ar_mask),
            "bg_de_mean": float(de_mean),
            "bg_de_frac": float(frac),
            "is_object": True,
            "registration": None
        })
        t_loop += (time.time() - t_l0) * 1000

    IOU_THR_DEDUP = 0.15  #0.20
    rects_dedup, pruned = dedup_overlapping_rects(rects_out, iou_thr=IOU_THR_DEDUP)
    rects_dedup, pruned = dedup_overlapping_rects(rects_dedup, iou_thr=IOU_THR_DEDUP)
    rects_dedup, pruned = dedup_overlapping_rects(rects_dedup, iou_thr=IOU_THR_DEDUP)

    # 增補：為最終輸出附上四個角點 (像素座標，int)
    for r in rects_dedup:
        cvrect = _rect_dict_to_cvtuple(r["rect"])
        pts = cv2.boxPoints(cvrect).astype(int).tolist()
        r["rect_corners"] = pts

    # 註冊
    t_reg = 0.0
    t_db = 0.0


    print("register ", register)
    if register:
        # if len(rects_dedup)>1:
        #     return {"ok": False, "error": "偵測出多個餐盤，註冊時只能單一餐盤。","objects": rects_dedup}
        # DB
        t0 = time.time()
        db_descs = build_db_descriptors(db_dir) if (db_dir and os.path.isdir(db_dir)) else []
        t_db = (time.time() - t0) * 1000
        if not db_dir:
            raise ValueError("register=True 時必須提供 db_dir")
        for r in rects_dedup:
            t_r0 = time.time()
            mi = r.get("mask_index")
            s = support_data.get(mi)
            if s is None:
                t_reg += (time.time() - t_r0) * 1000
                continue
            rect = s["rect"]; cnt = s["cnt"]; mm = s["mask"]

            patch = crop_rotated_rect(img, rect, pad=4)

            mask_full = np.zeros_like(mm)
            cv2.drawContours(mask_full, [cnt], -1, 255, thickness=cv2.FILLED)

            (cxr, cyr), (wr, hr), angr = rect
            w0, h0 = int(round(wr)), int(round(hr))
            angle = angr
            if w0 < h0:
                angle = angr + 90.0
                w0, h0 = h0, w0
            M = cv2.getRotationMatrix2D((cxr, cyr), angle, 1.0)
            mask_rot = cv2.warpAffine(mask_full, M, (W, H), flags=cv2.INTER_NEAREST)
            mask_patch = cv2.getRectSubPix(mask_rot, (int(round(w0 + 8)), int(round(h0 + 8))), (cxr, cyr))
            mask_patch = (mask_patch > 0).astype(np.uint8) * 255

            reg_result = register_object(
                cand_img=patch, cand_cnt=contour_from_mask(mask_patch), cand_mask=mask_patch, cand_rect=rect,
                db_descs=db_descs,
                shape_thr=shape_thr, ar_tol=ar_tol, area_tol=area_tol,
                color_thr=color_thr, min_colored_frac=min_colored_frac,
                feat_ratio=feat_ratio, feat_min_matches=feat_min_matches,
                db_dir=db_dir, save_ext=save_ext
            )
            r["registration"] = reg_result

            if reg_result and reg_result.get("action") == "saved":
                db_descs = build_db_descriptors(db_dir)
            t_reg += (time.time() - t_r0) * 1000

    # 最終視覺化
    drawn = img.copy()
    for am in all_masks_out:
        rc = am["rect"]
        cvrect = _rect_dict_to_cvtuple(rc)
        box = cv2.boxPoints(cvrect).astype(np.int32)
        cv2.drawContours(drawn, [box], 0, (255, 0, 0), 2)
        cx, cy = int(rc["cx"]), int(rc["cy"])
        cv2.putText(drawn, f"#{am['mask_index']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    for r in rects_dedup:
        rc = r["rect"]
        cvrect = _rect_dict_to_cvtuple(rc)
        box = cv2.boxPoints(cvrect).astype(np.int32)
        cv2.drawContours(drawn, [box], 0, (0, 255, 0), 3)
        tag = "NEW" if (r.get("registration", {}) or {}).get("action") == "saved" \
              else ("EXIST" if (r.get("registration", {}) or {}).get("action") == "exist" else "OBJ")
        cx, cy = int(rc["cx"]), int(rc["cy"])
        cv2.putText(drawn, tag, (cx, max(0, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if roi_clamped:
        x, y, w_, h_ = roi_clamped
        cv2.rectangle(drawn, (x, y), (x + w_ - 1, y + h_ - 1), (0, 255, 255), 2)

    annotated_path = None
    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        cv2.imwrite(out_path, drawn)
        annotated_path = out_path

    annotated_b64 = None
    if return_image:
        ok, buf = cv2.imencode(".png", drawn)
        if ok:
            annotated_b64 = "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

    timing_ms = {
        "read_image": round(t_img, 2),
        "read_bg": round(t_bg, 2),
        "build_db": round(t_db, 2),
        "deltaE_map": round(t_de, 2),
        "loop_masks": round(t_loop, 2),
        "registration": round(t_reg, 2),
        "total": round((time.time() - t_all0) * 1000, 2)
    }
    if debug:
        print("timing_ms = ", timing_ms)
    meta = {
        "rects": rects_dedup,
        "all_masks": all_masks_out,
        "dedup": {"iou_thr": IOU_THR_DEDUP, "pruned": pruned}
    }
    if roi_clamped:
        x, y, w_, h_ = roi_clamped
        meta["roi"] = {"x": x, "y": y, "w": w_, "h": h_}

    return {
        "ok": True,
        "objects": rects_dedup,
        # "all_masks": all_masks_out,
        # "dedup": meta["dedup"],
        # "annotated_image_path": annotated_path,
        **({"annotated_image_base64": annotated_b64} if annotated_b64 is not None else {}),
        # "timing_ms": timing_ms,
    }

    
# ======================================
# 既有偵測/色差/工具 —— 保持原邏輯
# （藉由傳入 config 覆寫參數，不改演算法）
# ======================================

def deltaE_ciede94_vec(lab1, lab2, application_type='graphic arts'):
    if application_type == 'graphic arts':
        K1, K2, Kl, Kc, Kh = 0.045, 0.015, 1.0, 1.0, 1.0
    elif application_type == 'textiles':
        K1, K2, Kl, Kc, Kh = 0.048, 0.014, 2.0, 1.0, 1.0
    else:
        raise ValueError("Unknown application_type. Use 'graphic arts' or 'textiles'.")
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    delta_L = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_sq = delta_a**2 + delta_b**2 - delta_C**2
    delta_H = np.sqrt(np.clip(delta_H_sq, 0, None))
    Sl = 1.0
    Sc = 1.0 + K1 * C1
    Sh = 1.0 + K2 * C1
    delta_E94 = np.sqrt((delta_L / (Kl * Sl))**2 + (delta_C / (Kc * Sc))**2 + (delta_H / (Kh * Sh))**2)
    return delta_E94


def deltaE_ciede2000_vec(lab1, lab2, *, dark_boost=True, kL_min=0.55, L_pivot=25.0, power=2.0, kC=1.0, kH=1.0):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    avg_L = (L1 + L2) / 2
    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-10)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p*a1p + b1*b1)
    C2p = np.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p) / 2
    h1p = (np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = (np.degrees(np.arctan2(b2, a2p)) % 360)
    deltahp = h2p - h1p
    deltahp = deltahp - 360 * (deltahp > 180)
    deltahp = deltahp + 360 * (deltahp < -180)
    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp) / 2)
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    avg_Hp = (h1p + h2p + 360 * (np.abs(h1p - h2p) > 180)) / 2
    avg_Hp %= 360
    T = (1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + 0.24 * np.cos(np.radians(2 * avg_Hp)) + 0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - 0.20 * np.cos(np.radians(4 * avg_Hp - 63)))
    delta_ro = 30 * np.exp(-((avg_Hp - 275)/25)**2)
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7 + 1e-10))
    Sl = 1 + (0.015 * ((avg_L - 50)**2)) / np.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_ro)) * Rc
    if dark_boost:
        t = np.clip(avg_L / L_pivot, 0.0, 1.0)
        kL = kL_min + (1.0 - kL_min) * (t ** power)
    else:
        kL = 1.0
    deltaE = np.sqrt((delta_Lp / (kL * Sl))**2 + (delta_Cp / (kC * Sc))**2 + (delta_Hp / (kH * Sh))**2 + Rt * (delta_Cp / (kC * Sc)) * (delta_Hp / (kH * Sh)))
    return deltaE


def rotated_rect_iou(rect1, rect2):
    int_type, int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if int_type == cv2.INTERSECT_NONE or int_pts is None:
        return 0.0
    int_area = cv2.contourArea(int_pts)
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    union_area = area1 + area2 - int_area
    if union_area <= 0:
        return 0.0
    return float(int_area) / float(union_area)

# ===== 預設參數（未在請求提供時使用這些值） =====
DEFAULTS2 = dict(
    FOLDER_PATH   = "images",
    BACKGROUND    = "empty.jpg",
    OUTPUT_DIR    = "output",
    ROI           = (154, 79, 523 - 154, 425 - 79),
    MIN_AREA      = 3000,
    GRAY_THR      = 60,
    LAB_THR_255   = 55,
    DE_THR        = 22.0,
    IOU_THR       = 0.30,
    USE_DE2000    = True,
    USE_DE94      = False,
    DE94_MODE     = 'graphic arts',

    REGISTER_DB_DIR        = "db_objects",
    REGISTER_SAVE_EXT      = ".png",
    REG_SHAPE_THR          = 0.10,
    REG_AR_TOL             = 0.10,
    REG_AREA_TOL           = 0.20,
    REG_COLOR_THR          = 0.30,
    REG_MIN_COLORED_FRAC   = 0.02,
    REG_FEAT_RATIO         = 0.75,
    REG_FEAT_MIN_MATCHES   = 1,

    # hue signature 參數
    H_BINS = 9, S_MIN = 40, V_MIN = 30, MIN_COUNT = 10, V_BLACK = 20,

    # deltaE2000 暗部加權
    DARK_BOOST = True, KL_MIN = 0.55, L_PIVOT = 25.0, POWER = 2.0, KC = 1.0, KH = 1.0,

    # ORB 設定（維持原本用 ORB）
    ORB_NFEATURES = 500,
)

def _ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _parse_roi(val) -> Tuple[int, int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) == 4:
        x, y, w, h = [int(v) for v in val]
        return (x, y, w, h)
    if isinstance(val, dict):
        x = int(val.get("x", 0)); y = int(val.get("y", 0))
        w = int(val.get("w", 0)); h = int(val.get("h", 0))
        return (x, y, w, h)
    # 字串 "x,y,w,h"
    if isinstance(val, str):
        xs = [int(v.strip()) for v in val.split(",")]
        if len(xs) == 4:
            return tuple(xs)  # type: ignore
    # fallback
    return DEFAULTS2["ROI"]

def build_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """把 request JSON 轉成內部參數（未給就用 DEFAULTS2）"""
    cfg = dict(DEFAULTS2)  # copy
    # 偵測區
    cfg["ROI"]         = _parse_roi(data.get("roi", DEFAULTS2["ROI"]))
    cfg["MIN_AREA"]    = int(data.get("min_area", DEFAULTS2["MIN_AREA"]))
    cfg["GRAY_THR"]    = int(data.get("gray_thr", DEFAULTS2["GRAY_THR"]))
    cfg["LAB_THR_255"] = int(data.get("lab_thr_255", DEFAULTS2["LAB_THR_255"]))
    cfg["DE_THR"]      = float(data.get("de_thr", DEFAULTS2["DE_THR"]))
    cfg["IOU_THR"]     = float(data.get("iou_thr", DEFAULTS2["IOU_THR"]))
    cfg["USE_DE2000"]  = bool(data.get("use_de2000", DEFAULTS2["USE_DE2000"]))
    cfg["USE_DE94"]    = bool(data.get("use_de94", DEFAULTS2["USE_DE94"]))
    cfg["DE94_MODE"]   = str(data.get("de94_mode", DEFAULTS2["DE94_MODE"]))

    # hue signature
    cfg["H_BINS"]      = int(data.get("h_bins", DEFAULTS2["H_BINS"]))
    cfg["S_MIN"]       = int(data.get("s_min", DEFAULTS2["S_MIN"]))
    cfg["V_MIN"]       = int(data.get("v_min", DEFAULTS2["V_MIN"]))
    cfg["MIN_COUNT"]   = int(data.get("min_count", DEFAULTS2["MIN_COUNT"]))
    cfg["V_BLACK"]     = int(data.get("v_black", DEFAULTS2["V_BLACK"]))

    # deltaE2000 暗部加權
    cfg["DARK_BOOST"]  = bool(data.get("dark_boost", DEFAULTS2["DARK_BOOST"]))
    cfg["KL_MIN"]      = float(data.get("kL_min", DEFAULTS2["KL_MIN"]))
    cfg["L_PIVOT"]     = float(data.get("L_pivot", DEFAULTS2["L_PIVOT"]))
    cfg["POWER"]       = float(data.get("power", DEFAULTS2["POWER"]))
    cfg["KC"]          = float(data.get("kC", DEFAULTS2["KC"]))
    cfg["KH"]          = float(data.get("kH", DEFAULTS2["KH"]))

    # 註冊區
    cfg["REGISTER_DB_DIR"]      = str(data.get("db_dir", DEFAULTS2["REGISTER_DB_DIR"]))
    cfg["REGISTER_SAVE_EXT"]    = str(data.get("register_save_ext", DEFAULTS2["REGISTER_SAVE_EXT"]))
    cfg["REG_SHAPE_THR"]        = float(data.get("reg_shape_thr", DEFAULTS2["REG_SHAPE_THR"]))
    cfg["REG_AR_TOL"]           = float(data.get("reg_ar_tol", DEFAULTS2["REG_AR_TOL"]))
    cfg["REG_AREA_TOL"]         = float(data.get("reg_area_tol", DEFAULTS2["REG_AREA_TOL"]))
    cfg["REG_COLOR_THR"]        = float(data.get("reg_color_thr", DEFAULTS2["REG_COLOR_THR"]))
    cfg["REG_MIN_COLORED_FRAC"] = float(data.get("reg_min_colored_frac", DEFAULTS2["REG_MIN_COLORED_FRAC"]))
    cfg["REG_FEAT_RATIO"]       = float(data.get("reg_feat_ratio", DEFAULTS2["REG_FEAT_RATIO"]))
    cfg["REG_FEAT_MIN_MATCHES"] = int(data.get("reg_feat_min_matches", DEFAULTS2["REG_FEAT_MIN_MATCHES"]))

    # I/O
    cfg["OUTPUT_DIR"]   = str(data.get("output_dir", DEFAULTS2["OUTPUT_DIR"]))

    # ORB
    cfg["ORB_NFEATURES"] = int(data.get("orb_nfeatures", DEFAULTS2["ORB_NFEATURES"]))
    return cfg

def load_background_from_path(bg_path: str, ROI: Tuple[int,int,int,int]):
    img_empty = cv2.imread(bg_path)
    if img_empty is None:
        raise FileNotFoundError(f"❌ 找不到背景圖：{bg_path}")
    x0, y0, w, h = ROI
    roi_empty = img_empty[y0:y0+h, x0:x0+w]
    gray_empty = cv2.cvtColor(roi_empty, cv2.COLOR_BGR2GRAY)
    lab_empty_u8 = cv2.cvtColor(roi_empty, cv2.COLOR_BGR2Lab)
    lab_empty = lab_empty_u8.astype(np.float32)
    lab_empty[..., 0] *= (100.0 / 255.0)
    lab_empty[..., 1:] -= 128.0
    return gray_empty, lab_empty_u8, lab_empty

def compute_masks(roi_img, gray_empty, lab_empty_u8, lab_empty, cfg):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(gray, gray_empty)
    _, mask_gray = cv2.threshold(diff_gray, cfg["GRAY_THR"], 255, cv2.THRESH_BINARY)

    lab_u8 = cv2.cvtColor(roi_img, cv2.COLOR_BGR2Lab)
    L1, a1, b1 = cv2.split(lab_empty_u8)
    L2, a2, b2 = cv2.split(lab_u8)
    lab_l2 = np.sqrt((L1.astype(np.float32) - L2.astype(np.float32))**2 +
                     (a1.astype(np.float32) - a2.astype(np.float32))**2 +
                     (b1.astype(np.float32) - b2.astype(np.float32))**2)
    _, mask_lab_simple = cv2.threshold(lab_l2.astype(np.uint8), cfg["LAB_THR_255"], 255, cv2.THRESH_BINARY)

    mask_lab_deltaE = None
    if cfg["USE_DE2000"] or cfg["USE_DE94"]:
        lab_target = cv2.cvtColor(roi_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_target[..., 0] *= (100.0 / 255.0)
        lab_target[..., 1:] -= 128.0
        if cfg["USE_DE2000"]:
            deltaE = deltaE_ciede2000_vec(
                lab_empty, lab_target,
                dark_boost=cfg["DARK_BOOST"], kL_min=cfg["KL_MIN"], L_pivot=cfg["L_PIVOT"],
                power=cfg["POWER"], kC=cfg["KC"], kH=cfg["KH"]
            )
        else:
            deltaE = deltaE_ciede94_vec(lab_empty, lab_target, application_type=cfg["DE94_MODE"])
        mask_lab_deltaE = (deltaE > cfg["DE_THR"]).astype(np.uint8) * 255

    mask_color = mask_lab_deltaE if (cfg["USE_DE2000"] or cfg["USE_DE94"]) else mask_lab_simple
    mask_and = cv2.bitwise_or(mask_gray, mask_color)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask_and = cv2.morphologyEx(mask_and, cv2.MORPH_OPEN, k1, iterations=1)
    mask_and = cv2.morphologyEx(mask_and, cv2.MORPH_ERODE, k1, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_and = cv2.morphologyEx(mask_and, cv2.MORPH_CLOSE, k2, iterations=1)

    region = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), dtype=np.uint8)
    mask_only_color = cv2.subtract(mask_color, mask_and)
    mask_only_gray = cv2.subtract(mask_gray, mask_and)
    region[..., 1] = mask_and
    region[..., 2] = mask_only_color
    region[..., 0] = mask_only_gray
    return mask_and, region

def find_boxes(mask_and, x0, y0, min_area):
    contours, _ = cv2.findContours(mask_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        # rect = rect_longside_as_width(rect)   #rico 轉換
        box = cv2.boxPoints(rect).astype(np.intp)
        box[:, 0] += x0
        box[:, 1] += y0
        center = (rect[0][0] + x0, rect[0][1] + y0)
        rect_moved = (center, rect[1], rect[2])
        candidates.append((area, box, rect_moved))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates

def overlay_result(img, region_roi, boxes, x0, y0):
    overlay = img.copy()
    roi_canvas = np.zeros_like(img, dtype=np.uint8)
    roi_canvas[y0:y0+region_roi.shape[0], x0:x0+region_roi.shape[1]] = region_roi
    cv2.addWeighted(roi_canvas, 0.4, overlay, 0.6, 0.0, overlay)
    for _, box, _ in boxes:
        cv2.drawContours(overlay, [box], 0, (0, 255, 255), 3)
    return overlay

def _split_bgra(img_any: np.ndarray):
    if img_any is None:
        return None, None
    if img_any.ndim == 3 and img_any.shape[2] == 4:
        return img_any[:, :, :3], img_any[:, :, 3]
    elif img_any.ndim == 3 and img_any.shape[2] == 3:
        return img_any, None
    elif img_any.ndim == 2:
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR), None
    else:
        return None, None

def _contour_from_mask(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def _minrect_metrics(cnt):
    if cnt is None:
        return None, 0.0, np.nan
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    area = float(cv2.contourArea(cnt))
    ar = float(max(w, h) / max(1e-6, min(w, h))) if (w > 0 and h > 0) else np.nan
    return rect, area, ar

def _compute_nonwhite_hue_sig(img_bgr, obj_mask=None, *, cfg):
    # 使用 cfg 的 H_BINS/S_MIN/V_MIN/MIN_COUNT/V_BLACK
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    C = np.sqrt(a*a + b*b)
    h = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0

    obj = (obj_mask > 0) if obj_mask is not None else np.ones(L.shape, dtype=bool)
    denom = int(obj.sum())
    if denom <= 0:
        return None, 0.0

    L_black = float(cfg["V_BLACK"]) * (100.0 / 255.0)
    C_min = max(6.0, 0.22 * float(cfg["S_MIN"]))

    black = (L < L_black) & obj
    graywhite = (C < C_min) & (~black) & obj
    colored = (~black) & (~graywhite) & obj

    cnt_colored = int(colored.sum())
    cnt_graywhite = int(graywhite.sum())
    cnt_black = int(black.sum())
    total = cnt_colored + cnt_graywhite + cnt_black
    if total < cfg["MIN_COUNT"]:
        colored_frac = cnt_colored / max(1.0, float(denom))
        return None, colored_frac

    h32 = h.astype(np.float32)
    mask_u8 = (colored.astype(np.uint8) * 255)
    hist = cv2.calcHist([h32], [0], mask_u8, [int(cfg["H_BINS"])], [0, 360]).astype(np.float32).flatten()
    vec = np.concatenate([hist, [float(cnt_graywhite), float(cnt_black)]], axis=0)
    sig = vec / (vec.sum() + 1e-6)
    colored_frac = cnt_colored / float(denom)
    return sig.tolist(), colored_frac

def _hue_distance(sig1, sig2):
    a = np.asarray(sig1, np.float32).reshape(-1, 1)
    b = np.asarray(sig2, np.float32).reshape(-1, 1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

def _akaze_desc(img_bgr, mask=None, *, cfg):
    # 維持原邏輯：使用 ORB（名稱沿用）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(cfg["ORB_NFEATURES"]))
    kp, des = orb.detectAndCompute(gray, mask)
    return kp, des

def _feat_good_matches(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good += 1
    return good

def _build_db_descriptors(db_dir, *, cfg):
    db_descs = []
    if not os.path.isdir(db_dir):
        _ensure_dir(db_dir)
        return db_descs
    for fn in os.listdir(db_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        p = os.path.join(db_dir, fn)
        img_any = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img_any is None:
            continue
        img_bgr, alpha = _split_bgra(img_any)
        if img_bgr is None:
            continue

        if alpha is not None:
            obj_mask = (alpha > 0).astype(np.uint8) * 255
            cnt = _contour_from_mask(obj_mask)
        else:
            g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnt1 = _contour_from_mask(th1)
            cnt2 = _contour_from_mask(th2)
            cnt = cnt1 if (cnt1 is not None and (cnt2 is None or cv2.contourArea(cnt1) >= cv2.contourArea(cnt2))) else cnt2
            obj_mask = None
            if cnt is not None:
                obj_mask = np.zeros(img_bgr.shape[:2], np.uint8)
                cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if cnt is None:
            continue

        rect, area, ar = _minrect_metrics(cnt)
        hsig, cfrac = _compute_nonwhite_hue_sig(img_bgr, obj_mask=obj_mask, cfg=cfg)
        kp, des = _akaze_desc(img_bgr, mask=obj_mask, cfg=cfg)

        db_descs.append({
            "file": fn,
            "contour": cnt,
            "area": area,
            "ar": ar,
            "rect": rect,
            "h_sig": hsig,
            "colored_frac": cfrac,
            "kp": kp,
            "des": des,
        })
    return db_descs

def _crop_rotated_rect(img, rect, pad=4):
    (cx, cy), (w, h), ang = rect
    w0, h0 = int(round(w)), int(round(h))
    angle = ang
    if w0 < h0:
        angle = ang + 90.0
        w0, h0 = h0, w0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    patch = cv2.getRectSubPix(rotated, (max(1, int(round(w0+2*pad))), max(1, int(round(h0+2*pad)))), (cx, cy))
    return patch, M, (w0, h0)

def _register_object(cand_img, cand_cnt, cand_mask, cand_rect, db_descs, cfg, debug=False) -> Dict[str, Any]:
    rect, area_mask, ar_mask = _minrect_metrics(cand_cnt)
    hsig, cfrac = _compute_nonwhite_hue_sig(cand_img, obj_mask=cand_mask, cfg=cfg)
    kp, des = _akaze_desc(cand_img, mask=cand_mask, cfg=cfg)

    best = None
    existed = False
    existed_file = None
    existed_pass_detail = None

    for d in db_descs:
        d_shape = 1e9
        try:
            d_shape = float(cv2.matchShapes(cand_cnt, d["contour"], cv2.CONTOURS_MATCH_I3, 0.0))
        except cv2.error:
            pass
        d_ar    = abs(ar_mask   - d["ar"])   / max(d["ar"],   1e-6)
        d_area  = abs(area_mask - d["area"]) / max(d["area"], 1e-6)

        d_hue = None
        color_ok = True
        color_eval = False
        if (hsig is not None) and (d["h_sig"] is not None) and (cfrac >= cfg["REG_MIN_COLORED_FRAC"]) and (d["colored_frac"] >= cfg["REG_MIN_COLORED_FRAC"]):
            d_hue = _hue_distance(hsig, d["h_sig"])
            color_ok = (d_hue <= cfg["REG_COLOR_THR"])
            color_eval = True

        # good = _feat_good_matches(des, d["des"], ratio=cfg["REG_FEAT_RATIO"])
        # feat_ok = (good >= cfg["REG_FEAT_MIN_MATCHES"])

        shape_ok = (d_shape <= cfg["REG_SHAPE_THR"])
        ar_ok    = (d_ar    <= cfg["REG_AR_TOL"])
        area_ok  = (d_area  <= cfg["REG_AREA_TOL"])
        pass_all = shape_ok and ar_ok and area_ok and color_ok #and feat_ok

        score = d_shape + 0.5*d_ar + 0.25*d_area + (0.5*(d_hue if d_hue is not None else 0.0)) #- 0.05*good
        info = {
            "db_file": d["file"],
            "shape": float(d_shape),
            "ar": float(d_ar),
            "area": float(d_area),
            "hue": (None if d_hue is None else float(d_hue)),
            # "feat": int(good),
            "pass": bool(pass_all),
            "score": float(score),
            "pass_flags": {
                "shape_ok": bool(shape_ok),
                "ar_ok": bool(ar_ok),
                "area_ok": bool(area_ok),
                "color_ok": bool(color_ok if color_eval else True),
                # "feat_ok": bool(feat_ok),
                "pass_all": bool(pass_all)
            }
        }
        if (best is None) or (info["score"] < best["score"]):
            best = info
        if pass_all:
            existed = True
            existed_file = d["file"]
            existed_pass_detail = info
            break

    if existed:
        return {
            "action": "exist",
            "saved_file": None,
            "db_dir": os.path.abspath(cfg["REGISTER_DB_DIR"]),
            "ok": True,
            "detail": {
                "pass": existed_pass_detail["pass_flags"],
                "same_db_file": existed_file,
                "score": existed_pass_detail["score"]
            }
        }

    # 新樣本儲存
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"obj_{ts}"
    k = 0
    while True:
        name = f"{base}_{k}{cfg['REGISTER_SAVE_EXT']}" if k > 0 else f"{base}{cfg['REGISTER_SAVE_EXT']}"
        outp = os.path.join(cfg["REGISTER_DB_DIR"], name)
        if not os.path.exists(outp):
            break
        k += 1
    _ensure_dir(cfg["REGISTER_DB_DIR"])
    if cfg["REGISTER_SAVE_EXT"].lower() == ".png":
        alpha = (cand_mask > 0).astype(np.uint8) * 255
        bgr = cand_img.copy()
        bgr[alpha == 0] = 0
        bgra = np.dstack([bgr, alpha])
        cv2.imwrite(outp, bgra)
    else:
        bgr = cand_img.copy()
        bgr[(cand_mask == 0)] = 0
        cv2.imwrite(outp, bgr)

    return {
        "action": "saved",
        "saved_file": os.path.basename(outp),
        "db_dir": os.path.abspath(cfg["REGISTER_DB_DIR"]),
        "ok": True,
        "detail": {
            "pass": best["pass_flags"] if best else {
                "shape_ok": False, "ar_ok": False, "area_ok": False, "color_ok": False, "feat_ok": False, "pass_all": False
            },
            "same_db_file": best["db_file"] if best else None,
            "score": best["score"] if best else 0.0
        }
    }

# =========================
# 單張處理（以 cfg 驅動）
# =========================
def process_one_image(img_path: str, gray_empty, lab_empty_u8, lab_empty, *, do_register: bool, cfg: Dict[str, Any]):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ 無法讀取：{img_path}")
    x0, y0, w, h = cfg["ROI"]
    roi_img = img[y0:y0+h, x0:x0+w]
    mask_and, region = compute_masks(roi_img, gray_empty, lab_empty_u8, lab_empty, cfg)
    boxes = find_boxes(mask_and, x0, y0, cfg["MIN_AREA"])

    # 註冊準備
    db_descs = _build_db_descriptors(cfg["REGISTER_DB_DIR"], cfg=cfg) if do_register else []

    full_mask = np.zeros(img.shape[:2], np.uint8)
    full_mask[y0:y0+h, x0:x0+w] = mask_and

    objects = []
    for idx, (_, box, rect) in enumerate(boxes):
        # 角點（以整張圖座標系，float）
        corners_float = box.astype(np.float32).reshape(-1, 2)
        rect_corners = [{"x": int(px), "y": int(py)} for (px, py) in corners_float]
        rect_json = {
            "angle": float(rect[2]),
            "cx": float(rect[0][0]),
            "cy": float(rect[0][1]),
            "w": float(rect[1][0]),
            "h": float(rect[1][1]),
        }
        reg_json: Dict[str, Any] = {
            "action": None, "saved_file": None, "db_dir": os.path.abspath(cfg["REGISTER_DB_DIR"]), "ok": False,
            "detail": {"pass": {"ar_ok": False, "area_ok": False, "color_ok": False, "feat_ok": False, "pass_all": False, "shape_ok": False}, "same_db_file": None, "score": 0.0}
        }
        if do_register:
            poly = np.zeros_like(full_mask)
            cv2.fillConvexPoly(poly, box.astype(np.int32), 255)
            cand_mask_full = cv2.bitwise_and(full_mask, poly)

            patch, M, (w0, h0) = _crop_rotated_rect(img, rect, pad=4)
            mask_rot = cv2.warpAffine(cand_mask_full, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            mask_patch = cv2.getRectSubPix(mask_rot, (int(round(w0+8)), int(round(h0+8))), (rect[0][0], rect[0][1]))
            mask_patch = (mask_patch > 0).astype(np.uint8) * 255

            cand_cnt = _contour_from_mask(mask_patch)
            if cand_cnt is not None:
                reg_json = _register_object(
                    cand_img=patch,
                    cand_cnt=cand_cnt,
                    cand_mask=mask_patch,
                    cand_rect=rect,
                    db_descs=db_descs,
                    cfg=cfg,
                    debug=False,
                )
                if reg_json.get("action") == "saved":
                    # 更新 DB，讓同張圖後續物件可用
                    db_descs = _build_db_descriptors(cfg["REGISTER_DB_DIR"], cfg=cfg)

        objects.append({
            "mask_index": int(idx),
            "rect": rect_json,
            "rect_corners": rect_corners,
            "registration": reg_json
        })
    return objects

# ---------- API ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True)

@app.route("/calibrate_z0", methods=["POST"])
def calibrate_z0():
    """
    使用校正圖計算 Z0 並寫入 JSON 檔。
    JSON:
    {
      "image_path": "img/calib.jpg",
      "z0_file": "./calibration/z0_config.json",     # 選填，預設 Z0_FILE_PATH
      "params": {
        "FOCAL_LENGTH_PIXEL": 960,
        "REAL_DIAGONAL_MM": 49.497,
        "REAL_DOT_LENGTH_MM": 35.0,
        "SELECT_MODE": "first4",
        "BLOB": { "minArea": 50, "maxArea": 2000, "minCircularity": 0.7 }
      },
      "z_offset_mm": 0.0  # 選填：若校正片與「容器基準面」有固定厚度差，可加上修正
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_path = data.get("image_path")
        if not image_path:
            return jsonify(error="缺少 image_path"), 400

        p = {**DEFAULTS, **(data.get("params") or {})}
        blob_cfg = {**DEFAULTS["BLOB"], **(p.get("BLOB") or {})}
        z0_file = os.path.abspath(data.get("z0_file") or Z0_FILE_PATH)
        z_offset = float(data.get("z_offset_mm") or 0.0)

        hinfo, code = compute_height_mm_from_image(
            image_path=str(image_path),
            REAL_DIAGONAL_MM=float(p["REAL_DIAGONAL_MM"]),
            REAL_DOT_LENGTH_MM=float(p["REAL_DOT_LENGTH_MM"]),
            FOCAL_LENGTH_PIXEL=float(p["FOCAL_LENGTH_PIXEL"]),
            blob_cfg=blob_cfg,
            select_mode=str(p.get("SELECT_MODE", "first4"))
        )
        if code != 200:
            return jsonify(hinfo), code

        # 以校正圖量得的 Z（加上可選 Offset）作為 Z0
        Z0 = float(hinfo["height_mm"]) + z_offset

        payload = dict(
            Z0=round(Z0, 3),
            updated_at=int(time.time()),
            meta=dict(
                z_offset_mm=round(z_offset, 3),
                image_path=os.path.abspath(image_path),
                focal_length_pixel=float(p["FOCAL_LENGTH_PIXEL"]),
                real_diagonal_mm=float(p["REAL_DIAGONAL_MM"]),
                real_dot_length_mm=float(p["REAL_DOT_LENGTH_MM"]),
                method="pinhole_avg(diagonal,side)",
                select_mode=str(p.get("SELECT_MODE", "first4")),
                blob=blob_cfg
            )
        )
        _atomic_write_json(z0_file, payload)
        return jsonify(ok=True, z0_file=z0_file, Z0=payload["Z0"]), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
        # return jsonify(error=str(e)), 500

@app.route("/measure", methods=["POST"])
def measure():
    """
    以 JSON 檔讀取 Z0 後進行量測。
    JSON:
    {
      "image_path": "img/2.jpg",
      "z0_file": "./calibration/z0_config.json",   # 選填，預設 Z0_FILE_PATH
      "params": {
        "FOCAL_LENGTH_PIXEL": 960,
        "REAL_DIAGONAL_MM": 49.497,
        "REAL_DOT_LENGTH_MM": 35.0,
        "SAVE_PREVIEW": true,
        "PREVIEW_PATH": "out/preview.jpg",
        "SELECT_MODE": "first4",
        "BLOB": { "minArea": 50, "maxArea": 2000, "minCircularity": 0.7 }
      }
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_path = data.get("image_path")
        if not image_path:
            return jsonify(error="缺少 image_path"), 400

        z0_file = os.path.abspath(data.get("z0_file") or Z0_FILE_PATH)
        Z0, z0_raw = _load_z0_from_file(z0_file)

        p = {**DEFAULTS, **(data.get("params") or {})}
        blob_cfg = {**DEFAULTS["BLOB"], **(p.get("BLOB") or {})}

        result, code = measure_from_image(
            image_path=str(image_path),
            Z0=float(Z0),
            REAL_DIAGONAL_MM=float(p["REAL_DIAGONAL_MM"]),
            REAL_DOT_LENGTH_MM=float(p["REAL_DOT_LENGTH_MM"]),
            FOCAL_LENGTH_PIXEL=float(p["FOCAL_LENGTH_PIXEL"]),
            blob_cfg=blob_cfg,
            select_mode=str(p.get("SELECT_MODE", "first4")),
            # save_preview=bool(p.get("SAVE_PREVIEW", True)),
            preview_path=p.get("PREVIEW_PATH")
        )
        # if code == 200:
            # result["z0_file"] = z0_file
            # result["z0_meta"] = z0_raw.get("meta", {})
        return jsonify(result), code
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
        # return jsonify(error=str(e)), 500

@app.route("/distortion_score", methods=["POST"])
def api_score():
    try:
        if not request.is_json:
            return jsonify({"error": "Please provide the parameters in application/json format."}), 400
        params = request.get_json()
        # 基本參數檢查
        for k in ["image_path", "cols", "rows"]:
            if k not in params:
                return jsonify({"ok": False, "error": f"Missing required parameter(s): {k}"}), 400

        report = compute_report(params)
        if debug:
            print(f"[distortion_score] report=", report)
        return jsonify(report)
    except FileNotFoundError as e:
        if debug:
            print(f"[distortion_score] FileNotFoundError err=", str(e))
        return jsonify({"ok": False, "error": str(e)}), 200
    except Exception as e:
        if debug:
            print(f"[distortion_score] Exception err=", str(e))
        # 回傳可診斷訊息（生產可改成更保守）
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}), 200        


@app.route("/set-standard", methods=["POST"])
def api_set_standard():
    data = request.get_json(silent=True) or {}
    roi = data.get("roi", [183,109,307,290])
    method = (data.get("method") or "lab_median").lower()

    use_camera = bool(data.get("use_camera", True))
    img = None
    captured_path = None
    print(data.get("image_path"))
    if use_camera or not data.get("image_path"):
        # 直接取像
        print("取像!!!")
        # camera_index = int(data.get("camera_index", DEFAULT_CAMERA_INDEX))
        # frame_w = int(data.get("frame_width", DEFAULT_FRAME_W))
        # frame_h = int(data.get("frame_height", DEFAULT_FRAME_H))
        save_captured = bool(data.get("save_captured", False))
        save_dir = data.get("save_dir") or "brightness_captures"
        image_ext = (data.get("image_ext") or "jpg").lstrip(".").lower()
        if image_ext not in {"jpg", "jpeg", "png", "bmp", "tiff"}:
            image_ext = "jpg"
        try:
            img = capture_frame()
            # img = capture_frame(camera_index=camera_index, frame_w=frame_w, frame_h=frame_h)
            if save_captured:
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                fname = f"setstandard_{ts}.{image_ext}"
                captured_path = os.path.join(save_dir, fname)
                cv2.imwrite(captured_path, img)
        except Exception as e:
            return jsonify({"ok": False, "error": f"mage capture failed: {e}"}), 200
    else:
        # 使用檔案路徑
        print("用檔案路徑!!!")
        image_path = data.get("image_path")
        if not os.path.exists(image_path):
            return jsonify({"ok": False, "error": "Please provide a valid image_path"}), 200
        img = cv2.imread(image_path)
        # img = cv2.resize(img, (640, 480))
        if img is None:
            return jsonify({"ok": False, "error": "Image loading failed."}), 200

    try:
        val = compute_brightness(img, roi=roi, method=method)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Brightness calculation failed: {e}"}), 200

    cfg = load_config()
    cfg.setdefault("meta", {})
    cfg["standard_brightness"] = val
    cfg["method"] = method
    cfg["updated_at"] = _now_iso()
    src = "camera" if (use_camera or not data.get("image_path")) else "file"
    cfg["meta"]["set_standard_source"] = src
    if captured_path:
        cfg["meta"]["set_standard_image_path"] = captured_path
    elif data.get("image_path"):
        cfg["meta"]["set_standard_image_path"] = os.path.abspath(data.get("image_path"))
    save_config(cfg)

    return jsonify({
        "ok": True,
        "standard_brightness": val,
        # "method": method,
        # "source": src,
        # "captured_image_path": captured_path,
        # "config_path": CONFIG_JSON
    })


@app.route("/auto-adjust", methods=["POST"])
def api_auto_adjust():
    # 載入標準亮度
    cfg = load_config()
    if "standard_brightness" not in cfg:
        return jsonify({"ok": False, "error": "Standard brightness has not been set. Please call the API /set-standard first."}), 200
    target = float(cfg["standard_brightness"])  # L* 0~100

    data = request.get_json(silent=True) or {}
    # camera_index = int(data.get("camera_index", DEFAULT_CAMERA_INDEX))
    # frame_w = int(data.get("frame_width", DEFAULT_FRAME_W))
    # frame_h = int(data.get("frame_height", DEFAULT_FRAME_H))
    # roi = data.get("roi")
    roi = data.get("roi", [183,109,307,290])
    tolerance = float(data.get("tolerance", 0.5))
    pwm_min = int(data.get("pwm_min", 0))
    pwm_max = int(data.get("pwm_max", 100))
    initial_pwm = data.get("initial_pwm")
    max_iters = int(data.get("max_iters", 15))
    settle_ms = int(data.get("settle_ms", 50))
    method = (data.get("method") or cfg.get("method") or "lab_median").lower()
    save_images = bool(data.get("save_images", False))
    save_dir = data.get("save_dir") or "brightness_captures"
    save_roi = bool(data.get("save_roi", False))
    image_ext = (data.get("image_ext") or "jpg").lstrip(".").lower()
    if image_ext not in {"jpg", "jpeg", "png", "bmp", "tiff"}:
        image_ext = "jpg"
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    if pwm_min >= pwm_max:
        return jsonify({"ok": False, "error": "pwm_min must be less than pwm_max."}), 200

    # 初始化二分邊界與起點
    left, right = pwm_min, pwm_max
    pwm = int(initial_pwm) if initial_pwm is not None else (left + right) // 2

    history: List[Dict[str, Any]] = []
    best_pwm = pwm
    best_brightness = None  # type: Optional[float]

    for it in range(1, max_iters + 1):
        fpath = None
        roi_path = None
        # 設定 PWM，等待穩定，再取像
        try:
            set_pwm(pwm)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "at_iter": it, "pwm": pwm}), 200

        time.sleep(max(0, settle_ms) / 1000.0)

        try:
            frame = capture_frame()
            curr = compute_brightness(frame, roi=roi, method=method)
        
            # 依需求保存當前影像與(可選)ROI
            if save_images:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # fname = f"iter{it:02d}_pwm{pwm}_L{curr:.2f}_{ts}.{image_ext}"
                fname = f"iter{it:02d}.{image_ext}"
                fpath = os.path.join(save_dir, fname)
                try:
                    cv2.imwrite(fpath, frame)
                except Exception:
                    fpath = None
                if roi:
                    try:
                        roi_img = _crop_roi(frame, roi)
                        # roi_fname = f"iter{it:02d}_pwm{pwm}_L{curr:.2f}_{ts}_roi.{image_ext}"
                        roi_fname = f"iter{it:02d}_ROI.{image_ext}"
                        roi_path = os.path.join(save_dir, roi_fname)
                        cv2.imwrite(roi_path, roi_img)
                    except Exception:
                        roi_path = None
        except Exception as e:
            return jsonify({"ok": False, "error": f"Image capture/brightness calculation failed: {e}", "at_iter": it}), 200

        history.append({"iter": it, "pwm": pwm, "brightness": curr, "image_path": fpath, "roi_path": roi_path})

        # 更新最佳解
        if best_brightness is None or abs(curr - target) < abs(best_brightness - target):
            best_brightness = curr
            best_pwm = pwm

        # 收斂判定
        if abs(curr - target) <= tolerance:
            break

        # 二分策略（單調假設）
        if curr < target:
            left = max(left, pwm + 1)
        else:
            right = min(right, pwm - 1)

        if left > right:
            # 單調性可能不完美，提早結束，採用目前最佳值
            break

        pwm = (left + right) // 2

    # # 寫入最佳 PWM
    # cfg.setdefault("meta", {})
    # cfg["best_pwm"] = int(best_pwm)
    # cfg["best_brightness"] = float(best_brightness if best_brightness is not None else float("nan"))
    # cfg["pwm_bounds"] = [pwm_min, pwm_max]
    # cfg["tolerance"] = tolerance
    # cfg["method"] = method
    # cfg["updated_at"] = _now_iso()
    # cfg.setdefault("history", []).extend(history)
    # save_config(cfg)

    return jsonify({
        "ok": True,
        "target": target,
        "best_pwm": int(best_pwm),
        "best_brightness": float(best_brightness if best_brightness is not None else float("nan")),
        "iterations": history,
        # "save_dir": save_dir if save_images else None,
        # "config_path": CONFIG_JSON
    })     

@app.post("/mask-match-register")
def api_mask_match_register():
    try:
        payload = request.get_json(force=True, silent=False)
        if not payload:
            return jsonify({"ok": False, "error": "empty JSON body"}), 400

        image_path = payload.get("image_path")
        bg_path = payload.get("bg_path")
        db_dir = payload.get("db_dir", "db_objects")
        register = bool(payload.get("register", False))

        

        if not image_path or not bg_path:
            return jsonify({"ok": False, "error": "image_path / bg_path 必填"}), 400
        if register and not db_dir:
            return jsonify({"ok": False, "error": "register=True 時必須提供 db_dir"}), 400
        
        # 先讀取影像尺寸以便調整 mask
        img0 = imread_safe(image_path)
        if img0 is None:
            return jsonify({"ok": False, "error": f"讀不到測試圖：{image_path}"}), 404
        H, W = img0.shape[:2]
        
        # 讀取 mask（pkl / 圖檔 / 內嵌）
        masks_arr = load_masks_from_request(payload, H, W)
           
        # ROI
        roi = payload.get("roi", [175,96,334,325])
        if roi is not None:
            if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
                return jsonify({"ok": False, "error": "roi 需為 [x,y,w,h]"}), 400
            roi = tuple(map(int, roi))

        # 門檻參數（若未給則用預設）
        params = {
            # 幾何/顏色門檻（用於註冊比對）
            "shape_thr": float(payload.get("shape_thr", 0.10)),
            "ar_tol": float(payload.get("ar_tol", 0.10)),
            "area_tol": float(payload.get("area_tol", 0.20)),
            "color_thr": float(payload.get("color_thr", 0.30)),
            "min_colored_frac": float(payload.get("min_colored_frac", 0.02)),
            # 背景 ΔE 判定
            "de_thr": float(payload.get("de_thr", 10.0)),
            "de_frac": float(payload.get("de_frac", 0.75)),

            "min_mask_area": int(payload.get("min_mask_area", 500)),
            "max_mask_area": int(payload.get("max_mask_area", 250000)),

            "solidity_thr": float(payload.get("solidity_thr", 0.9)),
            # 註冊控制
            "feat_ratio": float(payload.get("feat_ratio", 0.75)),
            "feat_min_matches": int(payload.get("feat_min_matches", 1)),
            "save_ext": str(payload.get("save_ext", ".png")),
        }

        result = process_single(
            image_path=image_path,
            bg_path=bg_path,
            db_dir=(db_dir or ""),
            masks_arr=masks_arr,
            roi=roi,
            register=register,
            out_path=payload.get("out_path", "mask_match_out/result.png"),
            return_image=bool(payload.get("return_image", False)),
            **params,
        )
        if register:
            result["db_dir"] = norm_dir(db_dir)
        result["params"] = params
        return jsonify(result)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/cv-detect-register', methods=['POST'])
def detect_register():
    if not request.is_json:
        return jsonify({"ok": False, "error": "請以 application/json 傳遞參數"}), 400
    data = request.get_json(silent=True) or {}

    do_register = bool(data.get('register', False))
    image_path = data.get('image_path')
    bg_path = data.get('bg_path')

    if not image_path:
        return jsonify({"ok": False, "error": "缺少必要參數 image_path"}), 400
    if not bg_path:
        return jsonify({"ok": False, "error": "缺少必要參數 bg_path"}), 400

    # 建立配置（允許請求覆寫）
    cfg = build_config(data)

    try:
        gray_empty, lab_empty_u8, lab_empty = load_background_from_path(bg_path, cfg["ROI"])
        objects = process_one_image(image_path, gray_empty, lab_empty_u8, lab_empty,
                                    do_register=do_register, cfg=cfg)

        # 產出結果圖
        _ensure_dir(cfg["OUTPUT_DIR"])
        basename = os.path.splitext(os.path.basename(image_path))[0]
        overlay = cv2.imread(image_path)
        x0, y0, w, h = cfg["ROI"]
        roi_img = overlay[y0:y0+h, x0:x0+w]
        mask_and, region = compute_masks(roi_img, gray_empty, lab_empty_u8, lab_empty, cfg)
        boxes = find_boxes(mask_and, x0, y0, cfg["MIN_AREA"])
        overlay_img = overlay_result(overlay, region, boxes, x0, y0)
        out_path = os.path.join(cfg["OUTPUT_DIR"], f"{basename}_marked.jpg")
        # cv2.imwrite(out_path, overlay_img)

        return jsonify({
            "ok": True, 
            "config_used": cfg,   # 回傳實際使用的參數，方便你追蹤
            "objects": objects,
            "result_image": out_path
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": f"內部錯誤: {e}"}), 200      


@app.route("/delete-latest-file", methods=["POST"])
def delete_latest():
    data = request.get_json(silent=True) or {}
    folder = data.get("dir", "db_objects")   # 預設 output 資料夾
    pattern = data.get("pattern", "*")   # 可選篩選檔案類型，例如 *.jpg

    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return jsonify({"ok": False, "error": "no files found"}), 404

    # 找出最後修改的檔案
    latest = max(files, key=os.path.getmtime)
    try:
        size = os.path.getsize(latest)
        os.remove(latest)
        return jsonify({
            "ok": True, 
            "deleted_file": os.path.basename(latest),
            "size_bytes": size,
            "folder": folder
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500         

@app.route("/delete-all-files", methods=["POST"])
def delete_all_files():
    data = request.get_json(silent=True) or {}
    folder = data.get("dir", "db_objects")
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": f"dir not found: {folder}"}), 404

    files = [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if os.path.isfile(p)]
    if not files:
        return jsonify({"ok": False, "error": "no files found"}), 404

    for p in files: os.remove(p)
    return jsonify({"ok": True, "folder": folder, "deleted_count": len(files)})

if __name__ == "__main__":
    # python app.py
    # 需求：pip install flask opencv-python numpy
    app.run(host="0.0.0.0", port=10030, debug=False)

""""
API 1: POST /api/brightness/set-standard
    - 目的：從已拍攝的影像計算「環境亮度」(L* in CIE Lab) 並寫入 JSON 當作標準亮度值。
    - 請求 JSON：
    {
    "image_path": "/abs/or/relative/path/to/image.png",
    "roi": [x, y, w, h] # 可選，計算亮度的區域(像素)
    "method": "lab_mean|lab_median" # 可選，預設 lab_mean
    }
    - 回應 JSON：{ "ok": true, "standard_brightness": 57.3, "method": "lab_mean", ... }


API 2: POST /api/brightness/auto-adjust
    - 目的：自動迭代調整 PWM 光源，讓影像亮度逼近標準亮度值，最後寫入最佳 PWM。
    - 影像由本服務以 OpenCV 直接取像（預設 /dev/video0 或 Windows 的 0 裝置）。
    - 單調性假設：PWM ↑ → 亮度 ↑（多數 LED 光源近似成立）。
    - 採用二分搜尋 + 追蹤最佳解，直到誤差在容許範圍或達到迭代上限。
    - 請求 JSON：
    {
    "camera_index": 0, # 可選，預設 0
    "frame_width": 640, # 可選
    "frame_height": 480, # 可選
    "roi": [x, y, w, h], # 可選
    "tolerance": 0.5, # 亮度容許誤差（L* 單位，0~100），預設 0.5
    "pwm_min": 0, # 可選，預設 0
    "pwm_max": 255, # 可選，預設 255
    "initial_pwm": null, # 可選，若不提供以中點開始
    "max_iters": 10, # 可選
    "settle_ms": 200, # 可選，設 PWM 後等待 ms 再取像
    "method": "lab_mean|lab_median" # 可選
    }
    - 回應 JSON：{ "ok": true, "target": 57.3, "best_pwm": 143, "best_brightness": 57.1, ... }


整合點：
- 本服務會呼叫您既有的 PWM 韌體 API： http://localhost:8765/fw/pwm/{channel}/{value}
其中 channel 預設為 1，可於下方常數調整。
- 標準亮度與最佳 PWM 皆寫入同一個 JSON。


執行方式：
pip install flask opencv-python requests
python brightness_control_api.py
# 服務預設在 http://127.0.0.1:5001


注意事項：
- OpenCV 的 Lab 轉換中 L 通道為 0~255，已換算回 L* 的 0~100 區間。
- 若取像裝置是 RealSense 或其他相機，請自行在 capture_frame() 替換成對應擷取流程。
- 若 PWM 與亮度不是嚴格單調，二分搜尋仍會返回當前觀測紀錄裡最接近目標的組合。
"""

"""
RESTful 版本：mask_match_rects_bg_register

用途（單筆，不做批次）：
- 以『空背景圖 vs 測試圖』的 Lab ΔE 判斷哪些 mask 內是真正的物件。
- 註冊流程（可選）：僅以 mask 範圍建立/比對描述子；新物件以 PNG(alpha=mask) 形式存入 DB。

HTTP 介面（Flask）：
- GET  /health                     -> 簡單健康檢查
- POST /mask-match-register        -> 主功能
  請求 JSON（必要欄位）：
    {
      "image_path": "path/to/test.jpg",
      "bg_path":    "path/to/empty.jpg",
      "db_dir":     "path/to/db",         # register=True 時必填
      "register":   true/false,

      # mask 資訊三擇一（優先順序由上而下）：
      "mask_pkl_path": "path/to/masks.pkl",   # 與舊工具相容
      "mask_paths": ["m1.png", "m2.png"],    # 0/255 或 0/1 的單通道圖
      "masks": {                                 # 直接內嵌（與 pkl 格式相近）
        "masks": [[...],[...],...],              # (n,H,W) 或 (H,W)
        "height": H, "width": W, "amount": n
      },

      # 可選：ROI 與輸出/回傳
      "roi": [x, y, w, h],
      "out_path": "out/annotated.png",         # 若給定則會寫檔
      "return_image": true                      # 若 true，回傳 base64 圖

      # 進階門檻（皆可省略，使用預設值）：
      "shape_thr": 0.10,
      "ar_tol": 0.10,
      "area_tol": 0.20,
      "color_thr": 0.40,
      "min_colored_frac": 0.02,
      "de_thr": 17.0,
      "de_frac": 0.80,
      "min_mask_area": 500,
      "feat_ratio": 0.75,
      "feat_min_matches": 1,
      "save_ext": ".png"
    }

回應 JSON：
  {
    "ok": true,
    "objects": [...],           # 去重後通過 ΔE 的物件（含註冊結果）
    "all_masks": [...],         # 所有通過 ROI 的 mask 基本資訊
    "dedup": {...},             # 去重細節
    "annotated_image_path": "... 或 null",
    "annotated_image_base64": "data:image/png;base64,...." 或省略,
    "timing_ms": {...},         # 各步驟耗時
    "params": {...}             # 最終採用的參數（便於追溯）
  }

相依：Python 3.8+, opencv-python, numpy, flask
啟動：
  pip install opencv-python numpy flask
  python mask_match_rects_bg_register_api.py

說明原則：
- 僅使用 mask 範圍做幾何/顏色/特徵的建立與比對（DB 亦支援 PNG alpha 當作 mask）。
- 以數字說話：回傳全部門檻與中間值，利於調參與定位問題。
"""