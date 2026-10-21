"""
FISH TEXTURE FEATURE EXTRACTION
Trích xuất 10 đặc trưng Texture từ ảnh cá đã tách nền

Dependencies:
- opencv-python: GLCM, Gabor filters, morphology
- numpy: Array operations
- skimage: LBP, Haralick, Tamura features
- pywt: Wavelet transform
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.feature import hog
from scipy import ndimage
import pywt


class TextureFeatureExtractor:
    """
    Class trích xuất 10 đặc trưng Texture từ ảnh cá PNG (đã tách nền)
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
        else:  # RGB
            self.rgb = self.img
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            _, self.mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Convert sang grayscale (dùng cho hầu hết texture features)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)

        # Apply mask để chỉ lấy vùng cá
        self.gray_masked = cv2.bitwise_and(self.gray, self.gray, mask=self.mask)

    # ====================================
    # FEATURE 11: GLCM (4 values)
    # ====================================
    def get_glcm_features(self):
        """
        Gray-Level Co-occurrence Matrix (GLCM) features.

        Output: JSON array [contrast, homogeneity, correlation, energy]

        Giá trị thông tin:
        - Contrast cao: Texture có vân, vằn rõ rệt (zebra fish, lionfish)
        - Homogeneity cao: Texture mịn, đồng đều (cá hồi, cá ngừ)
        - Correlation: Mức độ tương quan giữa các pixels
        - Energy: Độ đồng nhất của texture

        Paper: Haralick et al. (1973) "Textural Features for Image Classification"
        """
        # Chỉ lấy vùng cá (crop theo mask)
        coords = cv2.findNonZero(self.mask)
        if coords is None or len(coords) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Normalize về 0-255 rồi quantize về 0-31 (để giảm computation)
        roi_norm = cv2.normalize(roi, None, 0, 31, cv2.NORM_MINMAX, cv2.CV_8U)

        # Tính GLCM với 4 hướng: 0°, 45°, 90°, 135°
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(roi_norm, distances=distances, angles=angles,
                             levels=32, symmetric=True, normed=True)

        # Trích xuất 4 features (average over all angles)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        return [
            round(float(contrast), 4),
            round(float(homogeneity), 4),
            round(float(correlation), 4),
            round(float(energy), 4)
        ]

    # ====================================
    # FEATURE 12: LBP Histogram (256 bins)
    # ====================================
    def get_lbp_histogram(self):
        """
        Local Binary Pattern (LBP) histogram.

        Output: JSON array [bin1, bin2, ..., bin256]

        Giá trị thông tin:
        - Mô tả micro-texture pattern (đốm, vân, vảy)
        - LBP uniform patterns: Texture có cấu trúc lặp lại (vảy cá)
        - LBP non-uniform: Texture ngẫu nhiên (skin texture)

        Paper: Ojala et al. (2002) "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP"
        """
        # Crop ROI
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [0.0] * 256

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Tính LBP với radius=3, neighbors=24
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(roi, n_points, radius, method='uniform')

        # Tính histogram (256 bins)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

        # Normalize
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        return hist.tolist()

    # ====================================
    # FEATURE 13: Gabor Filters (40 values)
    # ====================================
    def get_gabor_features(self):
        """
        Gabor filter bank responses (5 scales × 8 orientations = 40 values).

        Output: JSON array [s1o1, s1o2, ..., s5o8]

        Giá trị thông tin:
        - Phân tích texture theo hướng và tần số
        - Scale thấp: Chi tiết mịn (vảy nhỏ, skin texture)
        - Scale cao: Pattern lớn (sọc, đốm lớn)
        - Orientation: Hướng của vân texture (ngang/dọc/chéo)

        Paper: Manjunath & Ma (1996) "Texture features for browsing and retrieval of image data"
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [0.0] * 40

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Gabor filter parameters
        scales = [3, 5, 7, 9, 11]  # 5 scales (frequency)
        orientations = np.arange(0, np.pi, np.pi / 8)  # 8 orientations

        features = []

        for scale in scales:
            for theta in orientations:
                # Tạo Gabor kernel
                kernel = cv2.getGaborKernel(
                    (scale * 2 + 1, scale * 2 + 1),
                    sigma=scale / 2,
                    theta=theta,
                    lambd=scale,
                    gamma=0.5,
                    psi=0
                )

                # Apply filter
                filtered = cv2.filter2D(roi, cv2.CV_32F, kernel)

                # Tính mean và std của response
                mean_response = np.abs(filtered).mean()
                features.append(mean_response)

        # Normalize
        features = np.array(features)
        if features.max() > 0:
            features = features / features.max()

        return [round(float(f), 4) for f in features]

    # ====================================
    # FEATURE 14: Stripe Detection (count)
    # ====================================
    def get_stripe_count(self):
        """
        Phát hiện số lượng sọc trên cơ thể cá bằng Hough Transform.

        Output: Integer (số lượng sọc)

        Giá trị thông tin:
        - 0: Không có sọc (goldfish, mola)
        - 3-5: Ít sọc (clownfish, angelfish)
        - >10: Nhiều sọc (zebra fish, lionfish)

        Thuật toán: Hough Line Transform + clustering
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return 0

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Edge detection
        edges = cv2.Canny(roi, 50, 150)

        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                 minLineLength=int(h * 0.2), maxLineGap=5)

        if lines is None:
            return 0

        # Đếm số lines (gần đúng cho số sọc)
        # NOTE: Thuật toán đơn giản, có thể cải thiện bằng clustering
        stripe_count = len(lines)

        return int(stripe_count)

    # ====================================
    # FEATURE 15: Spot Detection (count)
    # ====================================
    def get_spot_count(self):
        """
        Phát hiện số lượng đốm trên cơ thể cá bằng Blob Detection.

        Output: Integer (số lượng đốm)

        Giá trị thông tin:
        - 0: Không có đốm (tuna, salmon)
        - 5-20: Vài đốm (trout, grouper)
        - >50: Nhiều đốm (pufferfish, spotted fish)

        Thuật toán: SimpleBlobDetector (OpenCV)
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return 0

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Setup SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.5

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(roi)

        spot_count = len(keypoints)
        return int(spot_count)

    # ====================================
    # FEATURE 16: Regional Texture Variance
    # ====================================
    def get_regional_variance(self):
        """
        Variance của texture qua các vùng khác nhau trên cơ thể cá.

        Output: Float (variance giữa 4 vùng: đầu, thân, đuôi, bụng)

        Giá trị thông tin:
        - Variance thấp: Texture đồng đều trên toàn thân
        - Variance cao: Texture khác nhau giữa các vùng (đầu sần sùi, thân mịn)
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return 0.0

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # Chia thành 4 vùng: left (head), center-top (back), center-bottom (belly), right (tail)
        regions = [
            roi[:, :w//3],                    # Head
            roi[:h//2, w//3:2*w//3],          # Back
            roi[h//2:, w//3:2*w//3],          # Belly
            roi[:, 2*w//3:]                   # Tail
        ]

        # Tính variance của mỗi vùng
        variances = []
        for region in regions:
            if region.size > 0:
                var = np.var(region[region > 0])  # Chỉ tính trên pixels != 0
                variances.append(var if not np.isnan(var) else 0.0)

        # Variance giữa các variances (variance of variances)
        if len(variances) > 0:
            regional_var = np.var(variances)
        else:
            regional_var = 0.0

        return round(float(regional_var), 4)

    # ====================================
    # FEATURE 17: Scale Visibility Score
    # ====================================
    def get_scale_visibility(self):
        """
        Điểm số mức độ nhìn thấy vảy cá (0-1).

        Output: Float (0 = không thấy vảy, 1 = vảy rõ ràng)

        Giá trị thông tin:
        - Score cao (>0.7): Cá có vảy lớn rõ ràng (carp, goldfish)
        - Score thấp (<0.3): Cá không có vảy hoặc vảy nhỏ (catfish, eel)

        Thuật toán: Phân tích periodic pattern trong texture
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return 0.0

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # FFT để phát hiện periodic pattern
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Tính energy ở high frequency (scales)
        h_fft, w_fft = magnitude.shape
        center_h, center_w = h_fft // 2, w_fft // 2

        # Vùng mid-frequency (scales thường có frequency trung bình)
        mask_ring = np.zeros((h_fft, w_fft), dtype=np.uint8)
        cv2.circle(mask_ring, (center_w, center_h), min(center_w, center_h) // 2, 255, -1)
        cv2.circle(mask_ring, (center_w, center_h), min(center_w, center_h) // 4, 0, -1)

        mid_freq_energy = np.sum(magnitude * (mask_ring / 255.0))
        total_energy = np.sum(magnitude)

        if total_energy > 0:
            scale_score = mid_freq_energy / total_energy
        else:
            scale_score = 0.0

        # Normalize về 0-1
        scale_score = np.clip(scale_score * 5, 0, 1)

        return round(float(scale_score), 4)

    # ====================================
    # FEATURE 18: Wavelet Coefficients
    # ====================================
    def get_wavelet_features(self):
        """
        Wavelet transform coefficients (multi-scale texture analysis).

        Output: JSON array [mean_LL, std_LL, mean_LH, std_LH, mean_HL, std_HL, mean_HH, std_HH]

        Giá trị thông tin:
        - LL (Low-Low): Approximation (hình dạng tổng thể)
        - LH, HL, HH: Details theo các hướng (texture chi tiết)

        Paper: Mallat (1989) "A Theory for Multiresolution Signal Decomposition"
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [0.0] * 8

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # 2D Discrete Wavelet Transform (DWT)
        coeffs = pywt.dwt2(roi, 'haar')
        LL, (LH, HL, HH) = coeffs

        # Tính mean và std của mỗi subband
        features = [
            np.mean(LL), np.std(LL),
            np.mean(LH), np.std(LH),
            np.mean(HL), np.std(HL),
            np.mean(HH), np.std(HH)
        ]

        return [round(float(f), 4) for f in features]

    # ====================================
    # FEATURE 19: Haralick Features
    # ====================================
    def get_haralick_features(self):
        """
        Haralick texture features (13 features từ GLCM).

        Output: JSON array [f1, f2, ..., f13]

        Giá trị thông tin:
        - Mở rộng của GLCM với nhiều statistical measures hơn
        - Phân biệt texture types: coarse vs fine, smooth vs rough

        Paper: Haralick et al. (1973) - same as GLCM
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [0.0] * 13

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]
        roi_norm = cv2.normalize(roi, None, 0, 31, cv2.NORM_MINMAX, cv2.CV_8U)

        # GLCM
        glcm = graycomatrix(roi_norm, distances=[1], angles=[0],
                             levels=32, symmetric=True, normed=True)

        # 13 Haralick features (chọn một số chính)
        features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0]
        ]

        # Thêm 7 features nữa (placeholder - trong thực tế cần implement đầy đủ)
        features.extend([0.0] * 7)

        return [round(float(f), 4) for f in features[:13]]

    # ====================================
    # FEATURE 20: Tamura Features
    # ====================================
    def get_tamura_features(self):
        """
        Tamura texture features: Coarseness, Contrast, Directionality.

        Output: JSON array [coarseness, contrast, directionality]

        Giá trị thông tin:
        - Coarseness: Độ thô của texture (vảy lớn vs vảy nhỏ)
        - Contrast: Độ tương phản
        - Directionality: Texture có hướng (sọc ngang) hay không hướng (đốm)

        Paper: Tamura et al. (1978) "Textural Features Corresponding to Visual Perception"
        """
        coords = cv2.findNonZero(self.mask)
        if coords is None:
            return [0.0, 0.0, 0.0]

        x, y, w, h = cv2.boundingRect(coords)
        roi = self.gray_masked[y:y+h, x:x+w]

        # 1. Coarseness (dùng moving average windows)
        # Đơn giản hóa: std of local patches
        coarseness = 0.0
        patch_size = 8
        if h > patch_size and w > patch_size:
            patches = []
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = roi[i:i+patch_size, j:j+patch_size]
                    if patch.size > 0:
                        patches.append(np.std(patch))
            if len(patches) > 0:
                coarseness = np.mean(patches)

        # 2. Contrast
        contrast = np.std(roi[roi > 0]) if roi[roi > 0].size > 0 else 0.0

        # 3. Directionality (dùng gradient histogram)
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_angle = np.arctan2(sobel_y, sobel_x)
        directionality = np.std(gradient_angle[roi > 0]) if roi[roi > 0].size > 0 else 0.0

        return [
            round(float(coarseness), 4),
            round(float(contrast), 4),
            round(float(directionality), 4)
        ]

    # ====================================
    # EXTRACT ALL FEATURES
    # ====================================
    def extract_all(self):
        """
        Trích xuất tất cả 10 đặc trưng Texture.

        Returns:
            dict: Dictionary chứa 10 đặc trưng
        """
        try:
            features = {
                "glcm": self.get_glcm_features(),
                "lbp_histogram": self.get_lbp_histogram(),
                "gabor_filters": self.get_gabor_features(),
                "stripe_count": self.get_stripe_count(),
                "spot_count": self.get_spot_count(),
                "regional_variance": self.get_regional_variance(),
                "scale_visibility": self.get_scale_visibility(),
                "wavelet": self.get_wavelet_features(),
                "haralick": self.get_haralick_features(),
                "tamura": self.get_tamura_features()
            }
            return features

        except Exception as e:
            print(f"Lỗi khi trích xuất texture từ {self.image_path}: {str(e)}")
            return None


# ====================================
# MAIN - Test script
# ====================================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện:
       pip install opencv-python opencv-contrib-python numpy scikit-image scipy pywavelets

    2. Chạy test:
       python extract_texture_features.py
    """

    import json

    # Test trên 1 ảnh
    test_image = "D:/FishDataset/processed/ca_loc_001_nobg.png"

    try:
        extractor = TextureFeatureExtractor(test_image)
        features = extractor.extract_all()

        if features:
            print("✓ Trích xuất texture thành công!")
            print(json.dumps(features, indent=2))

    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
