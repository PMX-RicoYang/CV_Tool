import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
from statistics import median
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

def get_polygon_mask(shape, polygon_pts):
    """
    建立 polygon 遮罩
    shape: (H, W) -> 影像尺寸
    polygon_pts: list of [x, y]
    return: mask (uint8, 值為 0 或 1)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    polygon_np = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_np], 1)
    return mask

def visualize_delta_z_with_values(delta_z, step=50, min_display_value=1.0):
    """
    顯示 delta_z 熱度圖，並在部分像素上標註其數值
    step: 每隔多少像素顯示一次
    min_display_value: 最小值門檻（太小不顯示）
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(delta_z, cmap='jet')
    plt.colorbar(label="ΔZ (mm)")
    plt.title("delta_z Heatmap with Values")
    plt.xlabel("X")
    plt.ylabel("Y")

    h, w = delta_z.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            val = delta_z[y, x]
            if val >= min_display_value:
                plt.text(x, y, f"{val:.1f}", fontsize=6, color='white', ha='center', va='center')

    plt.tight_layout()
    plt.show()

def load_depth_from_yuy2_cv2(filepath, width, height, max_depth_mm=2000):
    """
    從 YUY2 (YUV422) 檔案讀取 Y 分量，轉為灰階深度圖
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    frame = np.frombuffer(raw_data, dtype=np.uint8)

    if frame.size != width * height * 2:
        raise ValueError(f"資料長度錯誤：期望 {width*height*2} bytes，實際 {frame.size} bytes")

    # YUY2 格式轉換成 (H, W, 2)
    yuy2_img = frame.reshape((height, width, 2))  # 每 pixel 2 bytes (YxUV)
    
    # 用 OpenCV 把 YUY2 轉成 BGR
    bgr_img = cv2.cvtColor(yuy2_img, cv2.COLOR_YUV2BGR_YUY2)

    # 再轉灰階（其實你也可以直接取 Y 分量）
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 轉成「深度圖（mm）」，假設灰階值代表 0~max_depth_mm
    depth_mm = (gray.astype(np.float32) / 255.0) * max_depth_mm

    return depth_mm, gray


def show_depth_image(gray_image, title="Gray Depth Image"):
    """
    顯示灰階深度圖
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(gray_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.colorbar(label="Pixel Intensity")
    plt.tight_layout()
    plt.show()    

def load_depth_from_yuy2(filepath, width, height, max_depth_mm=2000):
    """
    從 YUY2 (YUV422 packed) 讀取 Y 分量作為深度圖
    """
    expected_bytes = width * height * 2  # 每像素 2 bytes (YUYV for 2 pixels)
    with open(filepath, 'rb') as f:
        yuy2_data = f.read()

    if len(yuy2_data) != expected_bytes:
        raise ValueError(f"YUY2 檔案大小錯誤，期望 {expected_bytes} bytes，實際 {len(yuy2_data)} bytes")

    yuy2 = np.frombuffer(yuy2_data, dtype=np.uint8)
    y_vals = yuy2[::2]  # 每兩個 byte跳一個（取 Y0, Y1, Y2,...）

    if y_vals.size != width * height:
        raise ValueError("Y 分量大小不符合影像尺寸")

    y_image = y_vals.reshape((height, width))
    depth_mm = (y_image.astype(np.float32) / 255.0) * max_depth_mm
    # depth_mm = (1.0 - (y_image.astype(np.float32) / 255.0)) * max_depth_mm


    # depth_mm[depth_mm < 20] = 0     # 過近雜訊
    # depth_mm[depth_mm > 1500] = 0   # 異常高值雜訊

    return depth_mm

def load_depth_from_bmp(filepath, max_depth_mm=2000):
    """
    讀取 24-bit BMP 深度圖，轉為灰階後轉為深度（mm）
    """
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)  # shape=(H,W,3)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到檔案: {filepath}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 轉灰階
    depth_mm = (gray.astype(np.float32) / 255.0) * max_depth_mm
    return depth_mm




def fill_roi_holes_until_full(delta_z, roi_mask, max_iter=2):
    """
    僅補 ROI 中的 0 值，直到 ROI 區域內不再有 0 為止
    delta_z: np.ndarray, 原始深度差
    roi_mask: np.ndarray (0/1), ROI 遮罩
    max_iter: 最多補幾輪（避免無限迴圈）
    """
    filled = delta_z.copy()
    for i in range(max_iter):
        holes = (filled == 0) & (roi_mask == 1)
        if not np.any(holes):
            break

        # 只用 ROI 內部做局部平均補值（3x3）
        local_mean = scipy.ndimage.generic_filter(filled, np.nanmean, size=10, mode='mirror')
        filled[holes] = local_mean[holes]

    return filled

def fill_holes_with_roi_mean(delta_z, roi_mask):
    """
    用 ROI 內部的 delta_z 平均值補 ROI 中值為 0 的 pixel
    """
    filled = delta_z.copy()
    roi_valid_mask = (roi_mask == 1) & (delta_z > 0)

    if not np.any(roi_valid_mask):
        raise ValueError("⚠️ ROI 中沒有有效的 delta_z 可用來計算平均")

    mean_val = np.mean(delta_z[roi_valid_mask])
    fill_mask = (roi_mask == 1) & (delta_z == 0)
    filled[fill_mask] = mean_val

    return filled, mean_val    

def estimate_volume_simple(depth_obj, depth_ref, pixel_size_m=0.002, polygon_pts=None):
    """
    利用 Z 高度差 × 像素面積估算體積（無需內參，適用俯視拍攝）
    """
    mask = ""
    delta_z = depth_ref - depth_obj  # 基準 - 物體
    # delta_z = depth_obj - depth_ref  # 基準 - 物體
    delta_z[delta_z < 5] = 0  # 負值代表低於參考面，不計入
    delta_z[delta_z > 200] = 0  # 負值代表低於參考面，不計入



    if polygon_pts:
        mask = get_polygon_mask(delta_z.shape, polygon_pts)
        delta_z[mask == 0] = 0  # 過濾 ROI 外部像素

    # # 印出每一筆有效值
    # rows, cols = delta_z.shape
    # print("🔍 每個有效 pixel 的 delta_z（mm）如下：")
    # for y in range(rows):
    #     for x in range(cols):
    #         dz = delta_z[y, x]
    #         if dz > 0:
    #             print(f"(row={y}, col={x}) → delta_z = {dz:.2f} mm")

    # # 補洞
    # delta_z = fill_roi_holes_until_full(delta_z, mask)
    # delta_z, mean_val = fill_holes_with_roi_mean(delta_z, mask)

    # visualize_delta_z_with_values(delta_z, step=40, min_display_value=1.0)     

    voxel_volume = (pixel_size_m ** 2) * (delta_z / 1000.0)  # mm → m
    total_volume = np.sum(voxel_volume)
    print(f"\n原始體積： {round(float(total_volume * 1e6), 3) :.2f} cm³")
    return round(float(total_volume * 1e6 *0.017/3), 3) # m³ → cm³

@app.route('/getVolume_multiObj', methods=['POST']) 
def estimate_volume_multiObj():
    data = request.get_json()
    try:
        obj_paths = data['obj_path']  # 多個物件影像路徑 (list)
        ref_path = data['ref_path']
        polygon_pts = data['polygon_pts']
        pixel_size_m = 0.002

        # 載入參考深度圖
        depth_ref = load_depth_from_yuy2(ref_path, 1280, 720, max_depth_mm=2000)
        depth_ref = cv2.bilateralFilter(depth_ref, d=9, sigmaColor=75, sigmaSpace=75)

        volumes = []

        for path in obj_paths:
            print(f"\n▶️ 處理檔案：{path}")
            depth_obj = load_depth_from_yuy2(path, 1280, 720, max_depth_mm=2000)
            depth_obj = cv2.bilateralFilter(depth_obj, d=9, sigmaColor=75, sigmaSpace=75)

            # show_depth_image(depth_obj, title=f"Depth Map (mm) - {path}")
            vol = estimate_volume_simple(depth_obj, depth_ref, pixel_size_m, polygon_pts)
            print(f"📦 單一體積：{vol:.2f} cm³")
            volumes.append(vol)

        # 平均體積
        # avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
        avg_volume = median(volumes) if volumes else 0.0
        print(f"\n✅ 平均體積：約 {avg_volume:.2f} cm³")

        return jsonify({
            "volume_cm3": round(avg_volume, 3),
            "individual_volumes": [round(v, 3) for v in volumes],
            "count": len(volumes)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getVolume', methods=['POST'])
def estimate_volume():
    data = request.get_json()
    # if not data or 'polygon_pts' not in data or 'depth_obj' not in data :
    #     return jsonify({"error": "少參數"}), 400

    try:
        # === 使用範例 ===

        # 讀取檔案
        obj_path = data['obj_path']
        # obj_path = "Depth1_gray_8830046_9412D1_3p.bmp"
        ref_path = data['ref_path']#"Depth1_gray_8611343_8596D1_base.bmp"
        print(f"\n obj_path : {obj_path}")
        # 載入並轉為深度圖（單位：mm）
        depth_obj = load_depth_from_yuy2(obj_path,640,480, max_depth_mm=2000)
        depth_ref = load_depth_from_yuy2(ref_path,640,480, max_depth_mm=2000)

        # depth_obj = cv2.GaussianBlur(depth_obj, (5, 5), sigmaX=1)
        # depth_ref = cv2.GaussianBlur(depth_ref, (5, 5), sigmaX=1)

        depth_obj = cv2.bilateralFilter(depth_obj, d=9, sigmaColor=75, sigmaSpace=75)
        depth_ref = cv2.bilateralFilter(depth_ref, d=9, sigmaColor=75, sigmaSpace=75)

        # depth_mm, gray_img = load_depth_from_yuy2_cv2(obj_path, 640, 480)
        # # 顯示灰階影像
        # show_depth_image(gray_img, title="Grayscale View of Depth (Y channel)")
        # 如果要顯示真實深度圖
        # show_depth_image(depth_ref, title="REF Depth Map (mm)")
        show_depth_image(depth_obj, title="Depth Map (mm)")

        # ROI polygon_pts 為四個點組成的清單（像你滑鼠選點）
        polygon_pts = data['polygon_pts']#[[590, 340], [590, 420], [720, 420], [720, 340]]

        # 每 pixel 寬高（假設相機固定俯視拍攝）
        pixel_size_m = 0.002  # 2mm × 2mm

        # 體積估算
        volume_cm3 = estimate_volume_simple(depth_obj, depth_ref, pixel_size_m, polygon_pts)
        print(f"\n📦 預估體積：約 {volume_cm3:.2f} cm³")
        return jsonify({"volume_cm3": volume_cm3 })

    except Exception as e:
        return jsonify({"error": str(e)}), 500    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)

