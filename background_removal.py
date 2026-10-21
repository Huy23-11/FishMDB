
"""
FISH BACKGROUND REMOVAL - Tách nền ảnh cá

Các phương pháp:
1. U2Net: Deep learning, dùng model pretrained.
2. GrabCut: Tách foreground dựa trên vùng ROI.
3. Adaptive: Thresholding + Edge Detection.

Tối ưu:
- Khởi tạo U2Net session đúng một lần.
- Cho phép chọn CPU/GPU.
- Bỏ qua ảnh đã xử lý.
- Ghi nhận thời gian và thống kê kết quả.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from rembg import remove, new_session


# ============================================================
# CẤU HÌNH
# ============================================================

INPUT_FOLDER = Path(
    r"C:\Users\messi2015\FishRetrievalSystem\FishDataset\original"
)

# Dùng thư mục riêng để không trộn kết quả BRIA cũ với U2Net.
OUTPUT_FOLDER = Path(
    r"C:\Users\messi2015\FishRetrievalSystem\FishDataset\processed_u2net"
)

# Chọn: "u2net", "grabcut", hoặc "adaptive"
METHOD = "u2net"

# Tên model sử dụng khi METHOD = "u2net".
# "u2net": ưu tiên chất lượng.
# "u2netp": model nhẹ hơn, thường phù hợp khi cần tốc độ.
MODEL_NAME = "u2net"

# False: chạy CPU, tránh lỗi CUDA hiện tại.
# Chỉ đặt True sau khi đã cài CUDA/cuDNN tương thích.
USE_GPU = False

# Bỏ qua ảnh đầu ra đã tồn tại.
SKIP_EXISTING = True

# Các định dạng ảnh đầu vào được hỗ trợ.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# PHƯƠNG PHÁP 1: U2NET / U2NETP
# ============================================================

def remove_background_u2net(image_path, output_path, session):
    """
    Tách nền bằng model U2Net hoặc U2NetP.

    session được tạo một lần ở process_all_images()
    và tái sử dụng cho các ảnh tiếp theo.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    # Đọc dữ liệu ảnh gốc.
    input_data = image_path.read_bytes()

    # Tách nền bằng session đã khởi tạo.
    output_data = remove(
        input_data,
        session=session
    )

    # Lưu ảnh PNG có alpha channel trong suốt.
    output_path.write_bytes(output_data)

    return output_path


# ============================================================
# PHƯƠNG PHÁP 2: GRABCUT
# ============================================================

def remove_background_grabcut(image_path, output_path):
    """
    Tách nền bằng GrabCut.

    Giả định cá nằm tương đối gần giữa ảnh.
    Không cần model deep learning.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    # Khởi tạo mask và các mô hình màu của GrabCut.
    mask = np.zeros((h, w), dtype=np.uint8)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # ROI: vùng trung tâm ảnh, bỏ qua khoảng 10% viền.
    margin = 0.10

    x1 = int(w * margin)
    y1 = int(h * margin)
    x2 = int(w * (1 - margin))
    y2 = int(h * (1 - margin))

    rect = (x1, y1, x2 - x1, y2 - y1)

    # Phân đoạn foreground/background.
    cv2.grabCut(
        img,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Giữ các pixel được xác định là foreground
    # hoặc foreground có khả năng.
    mask2 = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype(np.uint8)

    # Làm sạch mask.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask2 = cv2.morphologyEx(
        mask2,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask2 = cv2.morphologyEx(
        mask2,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Giữ lại phần foreground.
    result = img * mask2[:, :, np.newaxis]

    # Thêm alpha channel để nền trong suốt.
    result_rgba = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2BGRA
    )

    result_rgba[:, :, 3] = mask2 * 255

    if not cv2.imwrite(str(output_path), result_rgba):
        raise IOError(f"Không lưu được ảnh: {output_path}")

    return output_path


# ============================================================
# PHƯƠNG PHÁP 3: ADAPTIVE THRESHOLDING + EDGE DETECTION
# ============================================================

def remove_background_adaptive(image_path, output_path):
    """
    Tách nền bằng Adaptive Thresholding và Canny.

    Chỉ phù hợp với một số ảnh có độ tương phản rõ.
    Kết quả phụ thuộc nhiều vào màu sắc và độ phức tạp
    của nền ảnh.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    # Chuyển ảnh sang grayscale.
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Tạo mask sơ bộ bằng Adaptive Thresholding.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Phát hiện đường biên.
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Kết hợp threshold và edges.
    combined = cv2.bitwise_or(binary, edges)

    # Làm liền các vùng bị đứt.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    closed = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    # Loại bỏ vùng nền kết nối với bốn góc ảnh.
    mask = closed.copy()

    corners = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
    ]

    for point in corners:
        cv2.floodFill(
            mask,
            None,
            point,
            0
        )

    # Đảo mask và chuyển về 0/1.
    mask = cv2.bitwise_not(mask)
    mask = (mask > 0).astype(np.uint8)

    # Áp dụng mask lên ảnh gốc.
    result = img * mask[:, :, np.newaxis]

    # Chuyển sang BGRA, thiết lập độ trong suốt.
    result_rgba = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2BGRA
    )

    result_rgba[:, :, 3] = mask * 255

    if not cv2.imwrite(str(output_path), result_rgba):
        raise IOError(f"Không lưu được ảnh: {output_path}")

    return output_path


# ============================================================
# XỬ LÝ TOÀN BỘ THƯ MỤC
# ============================================================

def process_all_images(
    input_folder,
    output_folder,
    method="u2net"
):
    """
    Xử lý tất cả ảnh trong thư mục.

    Với U2Net:
    - Khởi tạo session một lần.
    - Tái sử dụng session cho mọi ảnh.

    Nếu SKIP_EXISTING = True:
    - Bỏ qua ảnh đầu ra đã tồn tại và có dung lượng > 0.
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh đầu vào: {input_path}"
        )

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Đường dẫn đầu vào không phải thư mục: {input_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    method = method.lower().strip()

    # Lấy danh sách ảnh và sắp xếp để thứ tự ổn định.
    image_files = sorted(
        [
            file
            for file in input_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )

    total = len(image_files)

    print("\n" + "=" * 65)
    print(f"Thư mục đầu vào : {input_path}")
    print(f"Số ảnh tìm thấy : {total}")
    print(f"Phương pháp      : {method.upper()}")
    print(f"Thư mục đầu ra  : {output_path}")
    print("=" * 65)

    if total == 0:
        print("Không tìm thấy ảnh đầu vào.")
        return

    # Kiểm tra phương pháp được chọn.
    if method not in {"u2net", "grabcut", "adaptive"}:
        raise ValueError(
            f"Phương pháp không hợp lệ: {method}. "
            "Chọn u2net, grabcut hoặc adaptive."
        )

    # --------------------------------------------------------
    # QUAN TRỌNG: KHỞI TẠO MODEL CHỈ MỘT LẦN
    # --------------------------------------------------------

    session = None

    if method == "u2net":
        if USE_GPU:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            # Ép dùng CPU để tránh cố tải CUDA
            # khi các thư viện GPU chưa được cài đúng.
            providers = [
                "CPUExecutionProvider",
            ]

        print(f"\nĐang khởi tạo model: {MODEL_NAME}")
        print(f"Execution providers yêu cầu: {providers}")
        print("Lần đầu có thể cần tải model về máy.")

        init_start = time.perf_counter()

        session = new_session(
            MODEL_NAME,
            providers=providers
        )

        init_elapsed = time.perf_counter() - init_start

        print(
            f"Đã khởi tạo session trong "
            f"{init_elapsed:.2f} giây."
        )

        print("Bắt đầu xử lý ảnh...\n")

    # --------------------------------------------------------
    # VÒNG LẶP XỬ LÝ ẢNH
    # --------------------------------------------------------

    success_count = 0
    skipped_count = 0
    error_count = 0

    total_processing_time = 0.0

    for idx, img_file in enumerate(image_files, start=1):

        output_file = output_path / (
            f"{img_file.stem}_nobg.png"
        )

        # Bỏ qua ảnh đã xử lý.
        if (
            SKIP_EXISTING
            and output_file.is_file()
            and output_file.stat().st_size > 0
        ):
            skipped_count += 1

            print(
                f"[{idx}/{total}] BỎ QUA: "
                f"{img_file.name} "
                "(đầu ra đã tồn tại)"
            )

            continue

        print(
            f"\n[{idx}/{total}] Đang xử lý: "
            f"{img_file.name}"
        )

        start_time = time.perf_counter()

        try:
            if method == "u2net":
                remove_background_u2net(
                    img_file,
                    output_file,
                    session
                )

            elif method == "grabcut":
                remove_background_grabcut(
                    img_file,
                    output_file
                )

            elif method == "adaptive":
                remove_background_adaptive(
                    img_file,
                    output_file
                )

            elapsed = time.perf_counter() - start_time

            total_processing_time += elapsed
            success_count += 1

            print(
                f"  ✓ Đã lưu: {output_file.name}"
            )

            print(
                f"  Thời gian: {elapsed:.2f} giây"
            )

        except Exception as e:
            error_count += 1

            # Xóa file đầu ra lỗi nếu đã được tạo dở.
            try:
                if (
                    output_file.exists()
                    and output_file.stat().st_size == 0
                ):
                    output_file.unlink()
            except OSError:
                pass

            print(
                f"  ✗ LỖI khi xử lý {img_file.name}: {e}"
            )

    # --------------------------------------------------------
    # TỔNG KẾT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOÀN THÀNH XỬ LÝ DATASET")
    print("=" * 65)

    print(f"Tổng số ảnh tìm thấy : {total}")
    print(f"Thành công trong lượt: {success_count}")
    print(f"Bỏ qua ảnh cũ        : {skipped_count}")
    print(f"Lỗi                  : {error_count}")

    if success_count > 0:
        average_time = total_processing_time / success_count

        print(
            f"Thời gian trung bình : "
            f"{average_time:.2f} giây/ảnh"
        )

    print(
        f"Tổng thời gian xử lý : "
        f"{total_processing_time:.2f} giây"
    )

    print(f"Thư mục kết quả      : {output_path}")
    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        process_all_images(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            method=METHOD
        )

    except KeyboardInterrupt:
        print(
            "\nĐã dừng chương trình theo yêu cầu. "
            "Các ảnh đã lưu vẫn được giữ lại."
        )

    except Exception as e:
        print(f"\nLỖI CHƯƠNG TRÌNH: {e}")
        raise
