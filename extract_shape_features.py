"""
FISH SHAPE FEATURE EXTRACTION
Trích xuất 10 đặc trưng Shape từ ảnh cá đã tách nền

Dependencies:
- opencv-python: Contour, moments, geometric features
- numpy: Array operations
- scipy: Fourier Transform
"""

import cv2
import numpy as np
from scipy.fft import fft
from scipy.spatial import distance
import json


class ShapeFeatureExtractor:
    """
    Class trích xuất 10 đặc trưng Shape từ ảnh cá PNG (đã tách nền)
    """

    def __init__(self, image_path):
        """
        Args:
            image_path: Đường dẫn ảnh PNG đã tách nền (có alpha channel)
        """
        self.image_path = image_path

        # Đọc ảnh (IMREAD_UNCHANGED để giữ alpha channel)
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        # Tách alpha channel (mask)
        if self.img.shape[2] == 4:  # RGBA
            self.mask = self.img[:, :, 3]
        else:  # RGB (không có alpha)
            # Convert sang grayscale rồi threshold
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            _, self.mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Tìm contour lớn nhất (= contour con cá)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            raise ValueError(f"Không tìm thấy contour trong ảnh: {image_path}")

        self.contour = max(contours, key=cv2.contourArea)  # Contour lớn nhất

        # Tính moments một lần (dùng cho nhiều đặc trưng)
        self.moments = cv2.moments(self.contour)

    # ====================================
    # FEATURE 1: Aspect Ratio
    # ====================================
    def get_aspect_ratio(self):
        """
        Tỉ lệ dài/rộng của bounding box.

        Công thức: AR = width / height

        Giá trị thông tin:
        - AR cao (>4.0): Cá thân dài mảnh (lươn, dao, băng)
        - AR thấp (<2.0): Cá thân ngắn hoặc cao (bơn, mola)
        """
        x, y, w, h = cv2.boundingRect(self.contour)
        if h == 0:
            return 0.0
        aspect_ratio = w / h
        return round(aspect_ratio, 4)

    # ====================================
    # FEATURE 2: Compactness
    # ====================================
    def get_compactness(self):
        """
        Độ chặt khít của hình dạng.

        Công thức: Compactness = (Perimeter^2) / Area

        Giá trị thông tin:
        - Compactness thấp (~12-20): Cá thân tròn, đầy đặn (mola, pufferfish)
        - Compactness cao (>50): Cá thân dài mảnh hoặc có nhiều vây nhô ra (lionfish)

        NOTE: Hình tròn hoàn hảo có Compactness = 4π ≈ 12.57
        """
        area = cv2.contourArea(self.contour)
        perimeter = cv2.arcLength(self.contour, True)

        if area == 0:
            return 0.0

        compactness = (perimeter ** 2) / area
        return round(compactness, 4)

    # ====================================
    # FEATURE 3: Elongation
    # ====================================
    def get_elongation(self):
        """
        Độ kéo dài của hình dạng (dựa trên ellipse fitting).

        Công thức: Elongation = Major Axis / Minor Axis

        Giá trị thông tin:
        - Elongation cao (>3.0): Cá thân dài (lươn, băng, pipefish)
        - Elongation thấp (~1.0-1.5): Cá thân tròn (mola, discus)
        """
        if len(self.contour) < 5:  # fitEllipse cần ít nhất 5 điểm
            return 1.0

        ellipse = cv2.fitEllipse(self.contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)

        if minor_axis == 0:
            return 1.0

        elongation = major_axis / minor_axis
        return round(elongation, 4)

    # ====================================
    # FEATURE 4: Contour Points (100 points)
    # ====================================
    def get_contour_points(self, num_points=100):
        """
        Lấy mẫu 100 điểm đều nhau trên contour.

        Output: JSON array [x1, y1, x2, y2, ..., x100, y100]

        Giá trị thông tin:
        - Mô tả chi tiết hình dạng contour
        - Dùng để so sánh hình dạng bằng Shape Context hoặc Procrustes Analysis
        """
        # Resample contour thành num_points điểm đều
        contour_length = cv2.arcLength(self.contour, True)
        step = contour_length / num_points

        sampled_points = []
        accumulated_length = 0
        target_length = 0

        for i in range(len(self.contour) - 1):
            pt1 = self.contour[i][0]
            pt2 = self.contour[i + 1][0]
            segment_length = np.linalg.norm(pt2 - pt1)

            while target_length <= accumulated_length + segment_length and len(sampled_points) < num_points:
                # Interpolate điểm trên segment này
                ratio = (target_length - accumulated_length) / segment_length if segment_length > 0 else 0
                point = pt1 + ratio * (pt2 - pt1)
                sampled_points.append(point.tolist())
                target_length += step

            accumulated_length += segment_length

        # Đảm bảo đủ 100 điểm
        while len(sampled_points) < num_points:
            sampled_points.append(sampled_points[-1])

        # Flatten thành [x1, y1, x2, y2, ...]
        flattened = [coord for point in sampled_points for coord in point]
        return flattened[:num_points * 2]  # Đảm bảo đúng 200 số (100 điểm × 2)

    # ====================================
    # FEATURE 5: Spine Curve (50 points)
    # ====================================
    def get_spine_curve(self, num_points=50):
        """
        Đường cong xương sống cá (skeleton medial axis).

        Output: JSON array [x1, y1, x2, y2, ..., x50, y50]

        Giá trị thông tin:
        - Mô tả độ cong của cơ thể cá
        - Phân biệt cá thẳng (lươn) vs cá cong (seahorse) vs cá xoắn
        """
        # Skeleton bằng morphological thinning
        skeleton = cv2.ximgproc.thinning(self.mask)

        # Tìm điểm trên skeleton
        skeleton_points = np.column_stack(np.where(skeleton > 0))

        if len(skeleton_points) == 0:
            return [0.0] * (num_points * 2)

        # Resample thành num_points điểm
        indices = np.linspace(0, len(skeleton_points) - 1, num_points).astype(int)
        sampled = skeleton_points[indices]

        # Flatten thành [x1, y1, x2, y2, ...]
        # NOTE: OpenCV uses (row, col) = (y, x), cần swap
        flattened = []
        for point in sampled:
            flattened.extend([int(point[1]), int(point[0])])  # (x, y)

        return flattened

    # ====================================
    # FEATURE 6: Fourier Descriptors (20 coefficients)
    # ====================================
    def get_fourier_descriptors(self, num_coeffs=20):
        """
        Mô tả hình dạng bằng Fourier Transform của contour.

        Output: JSON array [c1, c2, ..., c20] (magnitudes của FFT coefficients)

        Giá trị thông tin:
        - Mô tả hình dạng ở tần số khác nhau
        - Coefficients đầu: hình dạng tổng thể (tròn/dài)
        - Coefficients sau: chi tiết (vây, đuôi)
        - Bất biến với translation, rotation, scaling (nếu normalize đúng)
        """
        # Chuyển contour thành complex numbers
        contour_complex = self.contour[:, 0, 0] + 1j * self.contour[:, 0, 1]

        # FFT
        fft_result = fft(contour_complex)

        # Lấy magnitude của num_coeffs coefficients đầu (bỏ DC component)
        magnitudes = np.abs(fft_result[1:num_coeffs + 1])

        # Normalize bằng magnitude của coefficient đầu tiên
        if magnitudes[0] > 0:
            magnitudes = magnitudes / magnitudes[0]

        return magnitudes.tolist()

    # ====================================
    # FEATURE 7: Hu Moments (7 values)
    # ====================================
    def get_hu_moments(self):
        """
        7 Hu Moments (bất biến với translation, rotation, scaling).

        Output: JSON array [h1, h2, h3, h4, h5, h6, h7]

        Giá trị thông tin:
        - h1: Độ đối xứng tổng thể
        - h2-h7: Các đặc trưng hình học bất biến
        - Dùng để so sánh hình dạng bất chấp góc xoay/tỉ lệ
        """
        hu_moments = cv2.HuMoments(self.moments).flatten()

        # Log transform để scale về phạm vi hợp lý
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments.tolist()

    # ====================================
    # FEATURE 8: Anatomical Landmarks (10 points)
    # ====================================
    def get_landmarks(self):
        """
        10 điểm đặc trưng trên cơ thể cá:
        1. Head tip (đầu)
        2. Tail tip (đuôi)
        3-4. Dorsal fin start/end (vây lưng)
        5-6. Pectoral fin left/right (vây ngực)
        7-8. Ventral fin left/right (vây bụng)
        9-10. Eye position left/right (mắt)

        Output: JSON array [x1, y1, x2, y2, ..., x10, y10]

        Giá trị thông tin:
        - Vị trí tương đối của các bộ phận
        - Dùng để so sánh anatomical structure
        """
        # Đơn giản hóa: detect các điểm extreme trên contour
        leftmost = tuple(self.contour[self.contour[:, :, 0].argmin()][0])
        rightmost = tuple(self.contour[self.contour[:, :, 0].argmax()][0])
        topmost = tuple(self.contour[self.contour[:, :, 1].argmin()][0])
        bottommost = tuple(self.contour[self.contour[:, :, 1].argmax()][0])

        # Giả định: head = leftmost, tail = rightmost
        # Dorsal = topmost, Ventral = bottommost
        # (Trong thực tế cần thuật toán phức tạp hơn với segmentation)

        # Tạm thời trả về 10 điểm duplicate (placeholder)
        landmarks = [
            leftmost,     # Head
            rightmost,    # Tail
            topmost,      # Dorsal start
            topmost,      # Dorsal end
            leftmost,     # Pectoral left
            rightmost,    # Pectoral right
            bottommost,   # Ventral left
            bottommost,   # Ventral right
            leftmost,     # Eye left (placeholder)
            rightmost     # Eye right (placeholder)
        ]

        # Flatten
        flattened = [coord for point in landmarks for coord in point]
        return flattened

    # ====================================
    # FEATURE 9: Fin Ratios (4 values)
    # ====================================
    def get_fin_ratios(self):
        """
        Tỉ lệ diện tích vây so với thân:
        1. Dorsal fin ratio (vây lưng)
        2. Tail fin ratio (vây đuôi)
        3. Pectoral fin ratio (vây ngực)
        4. Pelvic fin ratio (vây bụng)

        Output: JSON array [dorsal, tail, pectoral, pelvic]

        Giá trị thông tin:
        - Phân biệt cá có vây lớn (lionfish, betta) vs vây nhỏ (eel)
        - Cá đuôi dài (goldfish) vs đuôi ngắn (tuna)

        NOTE: Thuật toán phức tạp cần deep learning segmentation.
        Đây là placeholder đơn giản dựa trên convexity defects.
        """
        # Convex hull để tìm các vùng lõm (có thể là vây)
        hull = cv2.convexHull(self.contour, returnPoints=False)
        if len(hull) < 3:
            return [0.0, 0.0, 0.0, 0.0]

        defects = cv2.convexityDefects(self.contour, hull)

        if defects is None:
            return [0.0, 0.0, 0.0, 0.0]

        # Tính tổng diện tích của các defects (gần đúng cho fins)
        total_area = cv2.contourArea(self.contour)
        defect_areas = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            # d = distance từ farthest point đến hull (scaled by 256)
            defect_areas.append(d / 256.0)

        # Sắp xếp và lấy 4 defects lớn nhất
        defect_areas.sort(reverse=True)
        top4 = defect_areas[:4] if len(defect_areas) >= 4 else defect_areas + [0.0] * (4 - len(defect_areas))

        # Normalize bằng total area
        fin_ratios = [area / total_area if total_area > 0 else 0.0 for area in top4]

        return [round(r, 4) for r in fin_ratios]

    # ====================================
    # FEATURE 10: Head/Body Ratio
    # ====================================
    def get_head_body_ratio(self):
        """
        Tỉ lệ diện tích đầu so với toàn thân.

        Công thức: Head Ratio = Head Area / Total Area

        Giá trị thông tin:
        - Cá đầu to (grouper, catfish): ratio cao (~0.3-0.4)
        - Cá đầu nhỏ (eel, ribbon fish): ratio thấp (~0.1-0.15)

        NOTE: Thuật toán đơn giản: giả định 1/3 trái của bounding box là đầu.
        Trong thực tế cần semantic segmentation.
        """
        x, y, w, h = cv2.boundingRect(self.contour)

        # Tạo mask cho 1/3 trái (giả định là đầu)
        head_mask = np.zeros_like(self.mask)
        head_region = (x, y, w // 3, h)
        cv2.rectangle(head_mask, (head_region[0], head_region[1]),
                      (head_region[0] + head_region[2], head_region[1] + head_region[3]),
                      255, -1)

        # Tính diện tích giao giữa head_mask và fish mask
        head_area = cv2.countNonZero(cv2.bitwise_and(self.mask, head_mask))
        total_area = cv2.countNonZero(self.mask)

        if total_area == 0:
            return 0.0

        ratio = head_area / total_area
        return round(ratio, 4)

    # ====================================
    # EXTRACT ALL FEATURES
    # ====================================
    def extract_all(self):
        """
        Trích xuất tất cả 10 đặc trưng Shape.

        Returns:
            dict: Dictionary chứa 10 đặc trưng
        """
        try:
            features = {
                "aspect_ratio": self.get_aspect_ratio(),
                "compactness": self.get_compactness(),
                "elongation": self.get_elongation(),
                "contour_points": self.get_contour_points(),
                "spine_curve": self.get_spine_curve(),
                "fourier_descriptors": self.get_fourier_descriptors(),
                "hu_moments": self.get_hu_moments(),
                "landmarks": self.get_landmarks(),
                "fin_ratios": self.get_fin_ratios(),
                "head_body_ratio": self.get_head_body_ratio()
            }
            return features

        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng từ {self.image_path}: {str(e)}")
            return None


# ====================================
# MAIN - Test script
# ====================================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện:
       pip install opencv-python opencv-contrib-python numpy scipy

    2. Chạy test:
       python extract_shape_features.py
    """

    # Test trên 1 ảnh
    test_image = "D:/FishDataset/processed/ca_loc_001_nobg.png"

    try:
        extractor = ShapeFeatureExtractor(test_image)
        features = extractor.extract_all()

        if features:
            print("✓ Trích xuất thành công!")
            print(json.dumps(features, indent=2))

    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
