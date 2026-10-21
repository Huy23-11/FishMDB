
"""
FISH BACKGROUND REMOVAL - Tách nền ảnh cá
Xử lý ảnh cá có nền phức tạp, con cá không nổi bật

Phương pháp:
1. U2Net Deep Learning (model pretrained tách nền tốt nhất)
2. GrabCut + Interactive Segmentation (backup cho ảnh khó)
3. Morphological Operations để làm sạch mask
"""

import cv2
import numpy as np
from pathlib import Path
import os
from rembg import remove
from PIL import Image
import io

# ====================
# PHƯƠNG PHÁP 1: U2Net (KHUYÊN DÙNG - TỐT NHẤT)
# ====================
def remove_background_u2net(image_path, output_path):
    """
    Tách nền bằng U2Net deep learning model (pretrained).
    Đây là phương pháp TỐT NHẤT cho ảnh có nền phức tạp.

    Library: rembg (https://github.com/danielgatis/rembg)
    Model: U2Net pretrained on massive dataset
    """
    print(f"[U2Net] Đang xử lý: {image_path}")

    # Đọc ảnh
    with open(image_path, 'rb') as input_file:
        input_data = input_file.read()

    # Tách nền bằng U2Net
    output_data = remove(input_data)

    # Lưu ảnh đã tách nền (PNG với alpha channel)
    with open(output_path, 'wb') as output_file:
        output_file.write(output_data)

    print(f"[U2Net] ✓ Đã lưu: {output_path}")
    return output_path


# ====================
# PHƯƠNG PHÁP 2: GrabCut + Automatic ROI Detection
# ====================
def remove_background_grabcut(image_path, output_path):
    """
    Tách nền bằng GrabCut algorithm (OpenCV).
    Dùng khi U2Net không khả dụng hoặc kết quả chưa tốt.

    GrabCut: Interactive foreground extraction using iterated graph cuts
    Paper: https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf
    """
    print(f"[GrabCut] Đang xử lý: {image_path}")

    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    # Tạo mask và models cho GrabCut
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Định nghĩa ROI (Region of Interest)
    # Giả định con cá ở giữa ảnh, margin 10%
    margin = 0.1
    x1 = int(w * margin)
    y1 = int(h * margin)
    x2 = int(w * (1 - margin))
    y2 = int(h * (1 - margin))
    rect = (x1, y1, x2 - x1, y2 - y1)

    # Chạy GrabCut (5 iterations)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Tạo binary mask (0: background, 1: foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Morphological operations để làm sạch mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply mask lên ảnh gốc
    result = img * mask2[:, :, np.newaxis]

    # Convert sang RGBA để có alpha channel
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = mask2 * 255  # Alpha channel

    # Lưu ảnh
    cv2.imwrite(str(output_path), result_rgba)
    print(f"[GrabCut] ✓ Đã lưu: {output_path}")
    return output_path


# ====================
# PHƯƠNG PHÁP 3: Adaptive Thresholding + Edge Detection
# ====================
def remove_background_adaptive(image_path, output_path):
    """
    Tách nền bằng Adaptive Thresholding kết hợp Edge Detection.
    Dùng cho ảnh có contrast tốt giữa cá và nền.

    Thuật toán:
    1. Convert sang grayscale
    2. Adaptive Thresholding để tạo binary mask
    3. Canny Edge Detection để refine boundary
    4. Morphological operations để fill holes
    """
    print(f"[Adaptive] Đang xử lý: {image_path}")

    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    # Convert sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    # Kết hợp binary và edges
    combined = cv2.bitwise_or(binary, edges)

    # Morphological closing để fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Flood fill từ corners để remove nền
    mask = closed.copy()
    cv2.floodFill(mask, None, (0, 0), 0)
    cv2.floodFill(mask, None, (w-1, 0), 0)
    cv2.floodFill(mask, None, (0, h-1), 0)
    cv2.floodFill(mask, None, (w-1, h-1), 0)

    # Invert mask
    mask = cv2.bitwise_not(mask)
    mask = mask // 255  # Convert to 0/1

    # Apply mask
    result = img * mask[:, :, np.newaxis]

    # Convert sang RGBA
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = mask * 255

    cv2.imwrite(str(output_path), result_rgba)
    print(f"[Adaptive] ✓ Đã lưu: {output_path}")
    return output_path


# ====================
# AUTO PROCESSING - Xử lý toàn bộ thư mục
# ====================
def process_all_images(input_folder, output_folder, method='u2net'):
    """
    Xử lý tất cả ảnh trong thư mục.

    Args:
        input_folder: Thư mục chứa ảnh JPG gốc
        output_folder: Thư mục lưu ảnh đã tách nền (PNG)
        method: 'u2net', 'grabcut', hoặc 'adaptive'
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh JPG
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPG"))

    print(f"\n{'='*60}")
    print(f"Tìm thấy {len(image_files)} ảnh trong {input_folder}")
    print(f"Phương pháp: {method.upper()}")
    print(f"Output: {output_folder}")
    print(f"{'='*60}\n")

    # Chọn phương pháp
    if method == 'u2net':
        remove_func = remove_background_u2net
    elif method == 'grabcut':
        remove_func = remove_background_grabcut
    elif method == 'adaptive':
        remove_func = remove_background_adaptive
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")

    success_count = 0
    error_count = 0

    for idx, img_file in enumerate(image_files, 1):
        try:
            # Tạo tên file output (đổi .jpg thành .png)
            output_file = output_path / f"{img_file.stem}_nobg.png"

            print(f"\n[{idx}/{len(image_files)}]", end=" ")
            remove_func(str(img_file), str(output_file))
            success_count += 1

        except Exception as e:
            print(f"✗ LỖI: {img_file.name} - {str(e)}")
            error_count += 1

    print(f"\n{'='*60}")
    print(f"Hoàn thành!")
    print(f"✓ Thành công: {success_count}")
    print(f"✗ Lỗi: {error_count}")
    print(f"{'='*60}\n")


# ====================
# MAIN - Hướng dẫn sử dụng
# ====================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện:
       pip install opencv-python numpy pillow rembg

    2. Chạy script:
       python background_removal.py

    3. Chỉnh sửa đường dẫn bên dưới:
    """

    # ========== CHỈNH SỬA Ở ĐÂY ==========
    INPUT_FOLDER = r"C:\Users\messi2015\FishRetrievalSystem\FishDataset\original\flipped_Big Head Carp 15.jpg"     # Thư mục chứa ảnh JPG gốc
    OUTPUT_FOLDER = r"C:\Users\messi2015\FishRetrievalSystem\FishDataset\processed\flipped_Big Head Carp 15_nobg.png"   # Thư mục lưu ảnh PNG đã tách nền
    METHOD = "u2net"  # Chọn: 'u2net', 'grabcut', hoặc 'adaptive'
    # =====================================

    # Xử lý toàn bộ ảnh
    # process_all_images(INPUT_FOLDER, OUTPUT_FOLDER, method=METHOD)
    remove_background_u2net(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\n💡 GỢI Ý:")
    print("- U2Net: Tốt nhất, nhưng cần cài rembg và download model (~176MB)")
    print("- GrabCut: Không cần model, tốt cho ảnh có cá ở giữa")
    print("- Adaptive: Nhanh nhất, tốt cho ảnh có contrast cao")
    print("\nNếu kết quả chưa tốt, thử phương pháp khác!")
