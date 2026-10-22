"""
FISH COLOR FEATURE EXTRACTION
Trích xuất 10 đặc trưng Color từ ảnh cá đã tách nền

Dependencies:
- opencv-python: Color space conversion, histogram
- numpy: Array operations, statistics
- sklearn: K-means clustering
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats


class ColorFeatureExtractor:
    """
    Class trích xuất 10 đặc trưng Color từ ảnh cá PNG (đã tách nền)
    """

    def __init__(self, image_path):
        """
        Args:
            image_path: Đường dẫn ảnh PNG đã tách nền (có alpha channel)
        """
        self.image_path = image_path

        # Đọc ảnh
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        # Tách alpha channel (mask)
        if self.img.shape[2] == 4:  # RGBA
            self.rgb = self.img[:, :, :3]
            self.mask = self.img[:, :, 3]
            if self.mask.max() > 0 and self.mask.max() < 255:
                self.mask = cv2.normalize(self.mask, None, 0, 255, cv2.NORM_MINMAX)
            _, self.mask = cv2.threshold(self.mask, 0, 255, cv2.THRESH_BINARY)
        else:  # RGB
            self.rgb = self.img
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            _, self.mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # RGB image (OpenCV uses BGR, cần convert)
        self.rgb_cv = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)

        # HSV image
        self.hsv = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)

        # Lấy pixels của vùng cá (flatten)
        self.fish_pixels_rgb = self.rgb_cv[self.mask > 0]
        self.fish_pixels_hsv = self.hsv[self.mask > 0]

    # ====================================
    # FEATURE 21: RGB Histogram (768 bins)
    # ====================================
    def get_rgb_histogram(self):
        """
        Histogram của 3 kênh RGB (256 bins × 3 = 768 bins).

        Output: JSON array [r1, r2, ..., r256, g1, g2, ..., g256, b1, b2, ..., b256]

        Giá trị thông tin:
        - Phân bố màu sắc tổng thể
        - Cá màu đỏ: peaks ở R channel
        - Cá màu xanh: peaks ở B channel
        - Cá vàng: peaks ở R và G channels
        """
        if len(self.fish_pixels_rgb) == 0:
            return [0.0] * 768

        # Tính histogram cho từng channel
        hist_r, _ = np.histogram(self.fish_pixels_rgb[:, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(self.fish_pixels_rgb[:, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(self.fish_pixels_rgb[:, 2], bins=256, range=(0, 256))

        # Normalize
        total_pixels = len(self.fish_pixels_rgb)
        hist_r = hist_r.astype(float) / total_pixels
        hist_g = hist_g.astype(float) / total_pixels
        hist_b = hist_b.astype(float) / total_pixels

        # Concat thành 1 vector 768 chiều
        histogram = np.concatenate([hist_r, hist_g, hist_b])

        return histogram.tolist()

    # ====================================
    # FEATURE 22: K-means Dominant Colors (top 3)
    # ====================================
    def get_dominant_colors(self, k=3):
        """
        Top K màu sắc dominant bằng K-means clustering.

        Output: JSON array [
            {"rgb": [r1, g1, b1], "percentage": p1},
            {"rgb": [r2, g2, b2], "percentage": p2},
            {"rgb": [r3, g3, b3], "percentage": p3}
        ]

        Giá trị thông tin:
        - Cá đơn sắc: 1 màu chiếm >80% (goldfish toàn vàng)
        - Cá 2 màu: 2 màu dominant (clownfish: cam + trắng)
        - Cá nhiều màu: 3+ màu (rainbow fish, mandarin fish)
        """
        if len(self.fish_pixels_rgb) == 0:
            return [{"rgb": [0, 0, 0], "percentage": 0.0}] * k

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(self.fish_pixels_rgb)

        # Lấy centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)

        # Tính percentage của mỗi cluster
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        percentages = label_counts / len(labels)

        # Sắp xếp theo percentage giảm dần
        sorted_indices = np.argsort(percentages)[::-1]

        result = []
        for idx in sorted_indices:
            result.append({
                "rgb": colors[idx].tolist(),
                "percentage": round(float(percentages[idx]), 4)
            })

        return result

    # ====================================
    # FEATURE 23: Color Moments (9 values)
    # ====================================
    def get_color_moments(self):
        """
        Color moments: Mean, Standard Deviation, Skewness cho R, G, B.

        Output: JSON array [mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b]

        Giá trị thông tin:
        - Mean: Màu trung bình (dominant hue)
        - Std: Độ phân tán màu (đồng nhất vs nhiều màu)
        - Skewness: Độ lệch phân bố (cá có vùng sáng/tối rõ rệt)
        """
        if len(self.fish_pixels_rgb) == 0:
            return [0.0] * 9

        # Tính mean, std, skewness cho mỗi channel
        moments = []
        for channel in range(3):  # R, G, B
            pixels = self.fish_pixels_rgb[:, channel]
            mean = np.mean(pixels)
            std = np.std(pixels)
            skewness = stats.skew(pixels)

            moments.extend([mean, std, skewness])

        # Rearrange: [mean_r, mean_g, mean_b, std_r, ...]
        result = [
            moments[0], moments[3], moments[6],  # means
            moments[1], moments[4], moments[7],  # stds
            moments[2], moments[5], moments[8]   # skews
        ]

        return [round(float(m), 4) for m in result]

    # ====================================
    # FEATURE 24: HSV Histogram (308 bins)
    # ====================================
    def get_hsv_histogram(self):
        """
        Histogram của HSV (H: 180 bins, S: 64 bins, V: 64 bins).

        Output: JSON array [h1, ..., h180, s1, ..., s64, v1, ..., v64]

        Giá trị thông tin:
        - H (Hue): Màu sắc chính (đỏ/xanh/vàng/cam)
        - S (Saturation): Độ bão hòa (màu sắc rực rỡ vs nhạt)
        - V (Value): Độ sáng (cá sáng vs tối)
        """
        if len(self.fish_pixels_hsv) == 0:
            return [0.0] * 308

        # Tính histogram
        # OpenCV HSV: H (0-179), S (0-255), V (0-255)
        hist_h, _ = np.histogram(self.fish_pixels_hsv[:, 0], bins=180, range=(0, 180))
        hist_s, _ = np.histogram(self.fish_pixels_hsv[:, 1], bins=64, range=(0, 256))
        hist_v, _ = np.histogram(self.fish_pixels_hsv[:, 2], bins=64, range=(0, 256))

        # Normalize
        total_pixels = len(self.fish_pixels_hsv)
        hist_h = hist_h.astype(float) / total_pixels
        hist_s = hist_s.astype(float) / total_pixels
        hist_v = hist_v.astype(float) / total_pixels

        # Concat
        histogram = np.concatenate([hist_h, hist_s, hist_v])

        return histogram.tolist()

    # ====================================
    # FEATURE 25: Brightness Mean
    # ====================================
    def get_brightness_mean(self):
        """
        Độ sáng trung bình (V channel trong HSV).

        Output: Float (0-255)

        Giá trị thông tin:
        - Brightness cao (>180): Cá màu sáng (goldfish vàng, white koi)
        - Brightness trung bình (100-180): Cá màu thường
        - Brightness thấp (<100): Cá màu tối (black molly, dark catfish)
        """
        if len(self.fish_pixels_hsv) == 0:
            return 0.0

        brightness = np.mean(self.fish_pixels_hsv[:, 2])  # V channel
        return round(float(brightness), 4)

    # ====================================
    # FEATURE 26: Saturation Mean
    # ====================================
    def get_saturation_mean(self):
        """
        Độ bão hòa màu trung bình (S channel trong HSV).

        Output: Float (0-255)

        Giá trị thông tin:
        - Saturation cao (>180): Màu rực rỡ (clownfish, mandarin fish, betta)
        - Saturation trung bình (80-180): Màu thường
        - Saturation thấp (<80): Màu nhạt, xám (gray fish, silver fish)
        """
        if len(self.fish_pixels_hsv) == 0:
            return 0.0

        saturation = np.mean(self.fish_pixels_hsv[:, 1])  # S channel
        return round(float(saturation), 4)

    # ====================================
    # FEATURE 27: Regional Colors (4 zones)
    # ====================================
    def get_regional_colors(self):
        """
        Màu sắc trung bình của 4 vùng: head, back, belly, tail.

        Output: JSON array [
            {"region": "head", "rgb": [r, g, b]},
            {"region": "back", "rgb": [r, g, b]},
            {"region": "belly", "rgb": [r, g, b]},
            {"region": "tail", "rgb": [r, g, b]}
        ]

        Giá trị thông tin:
        - Phân biệt cá có màu gradient (đầu tối → đuôi sáng)
        - Phân biệt cá có màu đối xứng (lưng vs bụng)
        - Phân biệt cá có đuôi màu khác thân
        """
        if len(self.fish_pixels_rgb) == 0:
            return [{"region": r, "rgb": [0, 0, 0]} for r in ["head", "back", "belly", "tail"]]

        # Tìm bounding box
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [{"region": r, "rgb": [0, 0, 0]} for r in ["head", "back", "belly", "tail"]]

        x, y, w, h = cv2.boundingRect(coords)

        # Chia thành 4 vùng
        regions = {
            "head": self.rgb_cv[y:y+h, x:x+w//3],
            "back": self.rgb_cv[y:y+h//2, x+w//3:x+2*w//3],
            "belly": self.rgb_cv[y+h//2:y+h, x+w//3:x+2*w//3],
            "tail": self.rgb_cv[y:y+h, x+2*w//3:x+w]
        }

        result = []
        for region_name, region_img in regions.items():
            # Lấy pixels trong mask
            region_mask = self.mask[region_img.shape[0]:region_img.shape[0], region_img.shape[1]:region_img.shape[1]]

            # Tính mean color
            if region_img.size > 0:
                # Apply mask (chỉ lấy pixels của cá)
                masked_region = region_img.reshape(-1, 3)
                if len(masked_region) > 0:
                    mean_color = np.mean(masked_region, axis=0).astype(int)
                else:
                    mean_color = [0, 0, 0]
            else:
                mean_color = [0, 0, 0]

            result.append({
                "region": region_name,
                "rgb": mean_color.tolist()
            })

        return result

    # ====================================
    # FEATURE 28: Color Variance
    # ====================================
    def get_color_variance(self):
        """
        Variance tổng hợp của màu sắc (trong RGB space).

        Output: Float

        Giá trị thông tin:
        - Variance thấp: Màu đồng nhất (goldfish toàn vàng)
        - Variance cao: Màu đa dạng (koi nhiều màu, rainbow fish)
        """
        if len(self.fish_pixels_rgb) == 0:
            return 0.0

        # Tính variance của tất cả 3 channels
        variance = np.var(self.fish_pixels_rgb)
        return round(float(variance), 4)

    # ====================================
    # FEATURE 29: Dorsal-Ventral Contrast (Countershading)
    # ====================================
    def get_countershading(self):
        """
        Độ tương phản màu giữa lưng (dorsal) và bụng (ventral).

        Output: Float (ratio: brightness_belly / brightness_back)

        Giá trị thông tin:
        - Ratio > 1.2: Countershading rõ (lưng tối, bụng sáng - tuna, shark, dolphin fish)
        - Ratio ~ 1.0: Không countershading (goldfish, betta)
        - Ratio < 0.8: Reverse countershading (một số flatfish)

        NOTE: Countershading là cơ chế ngụy trang: lưng tối (nhìn từ trên), bụng sáng (nhìn từ dưới)
        """
        if len(self.fish_pixels_hsv) == 0:
            return 1.0

        # Tìm bounding box
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return 1.0

        x, y, w, h = cv2.boundingRect(coords)

        # Chia thành 2 nửa: trên (back) và dưới (belly)
        back_region = self.hsv[y:y+h//2, x:x+w]
        belly_region = self.hsv[y+h//2:y+h, x:x+w]

        # Lấy pixels trong mask
        back_mask = self.mask[y:y+h//2, x:x+w]
        belly_mask = self.mask[y+h//2:y+h, x:x+w]

        back_pixels = back_region[back_mask > 0]
        belly_pixels = belly_region[belly_mask > 0]

        if len(back_pixels) == 0 or len(belly_pixels) == 0:
            return 1.0

        # Tính brightness (V channel) trung bình
        back_brightness = np.mean(back_pixels[:, 2])
        belly_brightness = np.mean(belly_pixels[:, 2])

        if back_brightness == 0:
            return 1.0

        contrast_ratio = belly_brightness / back_brightness
        return round(float(contrast_ratio), 4)

    # ====================================
    # FEATURE 30: Hue Distribution Type
    # ====================================
    def get_hue_distribution_type(self):
        """
        Phân loại phân bố Hue: uniform, dominant, bimodal.

        Output: String ("uniform" / "dominant" / "bimodal")

        Giá trị thông tin:
        - "uniform": Nhiều màu đều nhau (rainbow fish, parrotfish)
        - "dominant": 1 màu chính (goldfish vàng, blue tang xanh)
        - "bimodal": 2 màu chính (clownfish cam+trắng, koi đỏ+trắng)
        """
        if len(self.fish_pixels_hsv) == 0:
            return "unknown"

        # Tính histogram của Hue (18 bins để giảm noise)
        hist_h, _ = np.histogram(self.fish_pixels_hsv[:, 0], bins=18, range=(0, 180))
        hist_h = hist_h.astype(float) / hist_h.sum()

        # Tìm peaks
        peaks = []
        for i in range(1, len(hist_h) - 1):
            if hist_h[i] > hist_h[i-1] and hist_h[i] > hist_h[i+1]:
                peaks.append((i, hist_h[i]))

        # Sắp xếp peaks theo height
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Phân loại
        if len(peaks) == 0:
            return "uniform"
        elif peaks[0][1] > 0.5:  # 1 peak chiếm >50%
            return "dominant"
        elif len(peaks) >= 2 and peaks[1][1] > 0.2:  # 2 peaks rõ rệt
            return "bimodal"
        else:
            return "uniform"

    # ====================================
    # EXTRACT ALL FEATURES
    # ====================================
    def extract_all(self):
        """
        Trích xuất tất cả 10 đặc trưng Color.

        Returns:
            dict: Dictionary chứa 10 đặc trưng
        """
        try:
            features = {
                "rgb_histogram": self.get_rgb_histogram(),
                "dominant_colors": self.get_dominant_colors(),
                "color_moments": self.get_color_moments(),
                "hsv_histogram": self.get_hsv_histogram(),
                "brightness_mean": self.get_brightness_mean(),
                "saturation_mean": self.get_saturation_mean(),
                "regional_colors": self.get_regional_colors(),
                "color_variance": self.get_color_variance(),
                "dorsal_ventral_contrast": self.get_countershading(),
                "hue_distribution_type": self.get_hue_distribution_type()
            }
            return features

        except Exception as e:
            print(f"Lỗi khi trích xuất color từ {self.image_path}: {str(e)}")
            return None


# ====================================
# MAIN - Test script
# ====================================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện:
       pip install opencv-python numpy scikit-learn scipy

    2. Chạy test:
       python extract_color_features.py
    """

    import json

    # Test trên 1 ảnh
    test_image = "D:/FishDataset/processed/ca_loc_001_nobg.png"

    try:
        extractor = ColorFeatureExtractor(test_image)
        features = extractor.extract_all()

        if features:
            print("✓ Trích xuất color thành công!")
            print(json.dumps(features, indent=2))

    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
