"""
FISH IMAGE SIMILARITY SEARCH
Tìm kiếm ảnh cá tương đồng dựa trên 30 đặc trưng (Shape + Texture + Color)
đã lưu trong PostgreSQL (relational-object database).

INPUT:  Đường dẫn 1 ảnh cá mới (PNG đã tách nền là tốt nhất, nhưng JPG/PNG
        thường cũng chạy được nhờ fallback threshold có sẵn trong 3 extractor)
OUTPUT: Top-K ảnh giống nhất trong dataset, sắp xếp GIẢM DẦN theo độ tương đồng

FLOW:
1. Trích xuất 30 đặc trưng của ảnh query bằng 3 class extractor đã có sẵn
   (ShapeFeatureExtractor, TextureFeatureExtractor, ColorFeatureExtractor)
2. Load toàn bộ feature_vectors từ database (JOIN fish_images, fish_species)
3. Với mỗi ảnh trong DB, tính similarity theo 3 nhóm: Shape, Texture, Color
   - Mỗi nhóm có 10 đặc trưng với KIỂU DỮ LIỆU khác nhau (scalar, histogram,
     vector cố định, tập điểm toạ độ, dữ liệu dạng list-of-dict, categorical)
     nên mỗi kiểu dùng 1 hàm so sánh phù hợp (xem phần DISTANCE/SIMILARITY).
   - Similarity của từng nhóm = trung bình có trọng số các sub-feature.
4. Overall similarity = 0.4*shape + 0.3*texture + 0.3*color
   (khớp với default weight trong bảng similarity_cache của schema)
5. Sắp xếp giảm dần theo overall_similarity, lấy Top-K
6. Ghi log vào bảng user_queries; nếu ảnh query đã có image_id trong DB
   (vd: đang tính hàng xóm gần nhất cho ảnh có sẵn) thì ghi thêm vào
   similarity_cache

GHI CHÚ QUAN TRỌNG VỀ ẢNH QUERY (đã đối chiếu với dữ liệu thật trong DB):
- Toàn bộ dataset trong feature_vectors được trích xuất từ ảnh ĐÃ TÁCH NỀN
  bằng U2Net (xem batch_processing.py: "input_folder: Thư mục chứa ảnh PNG
  đã tách nền", và background_removal.py ghi ra processed_u2net/*.png).
- Vì vậy ảnh query cũng PHẢI được tách nền bằng cùng phương pháp (U2Net)
  trước khi đưa vào 3 extractor, nếu không contour/mask sẽ bị tính trên
  gần như toàn bộ khung ảnh (do fallback threshold>10 trong extractor rất
  yếu với ảnh có nền thật), khiến shape/color/texture KHÔNG so sánh được
  với dataset. Hàm extract_query_features() bên dưới tự động phát hiện và
  tách nền ảnh query nếu cần (xem _looks_background_removed /
  _remove_background_for_query).

GHI CHÚ VỀ CÁC GIỚI HẠN (tương tự tinh thần các NOTE trong code gốc của nhóm):
- shape_landmarks KHÔNG được dùng để so sánh vì trong extract_shape_features.py
  các điểm này hiện là placeholder (nhiều điểm bị duplicate từ leftmost/rightmost/
  topmost/bottommost), chưa phải landmark giải phẫu thật sự.
- shape_contour_points và shape_spine_curve là toạ độ pixel thô, nên được so
  sánh bằng Procrustes Analysis (scipy.spatial.procrustes) để bất biến với
  translation/rotation/scale, thay vì so Euclidean trực tiếp (vốn sẽ sai vì
  2 ảnh khác kích thước/góc chụp).
- Các scalar feature (aspect_ratio, brightness, stripe_count, ...) có range
  rất khác nhau nên được min-max normalize theo toàn bộ pool (dataset + ảnh
  query) tại thời điểm query, qua class DatasetStats.
- Khi dataset lớn (>10,000 ảnh) nên cân nhắc pgvector hoặc ANN (FAISS/HNSW)
  như note trong database_schema.sql thay vì load toàn bộ vào RAM như ở đây.

Dependencies (đã có sẵn trong requirements.txt của project):
    psycopg2-binary, numpy, scipy, opencv-python, scikit-learn, scikit-image, pywavelets
"""

import json
import time
import tempfile
import argparse
from pathlib import Path

import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from scipy.spatial import procrustes as scipy_procrustes

from extract_shape_features import ShapeFeatureExtractor
from extract_texture_features import TextureFeatureExtractor
from extract_color_features import ColorFeatureExtractor

# Dùng chung thư viện tách nền với background_removal.py để đảm bảo ảnh query
# được tiền xử lý GIỐNG HỆT pipeline đã tạo ra dataset (processed_u2net/*.png).
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# ============================================================
# HELPERS: parse an toàn JSON field -> numpy array / scalar
# ============================================================
def to_array(value, default_len=0, fill=0.0):
    """Chuyển field JSON (list, hoặc string JSON, hoặc None) -> numpy array."""
    if value is None:
        return np.full(default_len, fill, dtype=float) if default_len else np.array([])
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=float)
    return arr


def safe_scalar(value, fill=0.0):
    return fill if value is None else float(value)


# ============================================================
# DISTANCE / SIMILARITY PRIMITIVES
# (mỗi hàm trả similarity trong khoảng 0-1, 1 = giống hệt)
# ============================================================
def euclidean_similarity(a, b, scale=None):
    """
    Euclidean distance -> similarity bằng 1 / (1 + distance/scale).
    scale mặc định = sqrt(số chiều) để distance không phụ thuộc độ dài vector.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    dist = np.linalg.norm(a - b)
    if scale is None:
        scale = np.sqrt(len(a)) if len(a) > 0 else 1.0
    return float(1.0 / (1.0 + dist / max(scale, 1e-9)))


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def histogram_similarity(h1, h2):
    """
    So sánh 2 histogram đã normalize (tổng ~1) bằng Histogram Intersection -
    phương pháp chuẩn để so sánh histogram (đặc biệt với LBP, RGB, HSV).
    """
    h1, h2 = np.asarray(h1, dtype=float), np.asarray(h2, dtype=float)
    if h1.shape != h2.shape or h1.size == 0:
        return 0.0
    intersection = np.minimum(h1, h2).sum()
    denom = min(h1.sum(), h2.sum())
    if denom <= 0:
        return 0.0
    return float(np.clip(intersection / denom, 0.0, 1.0))


def minmax_similarity(v1, v2, vmin, vmax):
    """So sánh 2 scalar đã biết range [vmin, vmax] của toàn bộ pool hiện tại."""
    rng = vmax - vmin
    if rng <= 1e-9:
        return 1.0
    d = abs(v1 - v2) / rng
    return float(np.clip(1.0 - d, 0.0, 1.0))


def procrustes_similarity(points_a, points_b, num_points):
    """
    So sánh 2 tập điểm (contour/spine) bất biến translation, rotation, scale
    bằng Procrustes Analysis — phù hợp vì 2 ảnh cá có thể khác kích thước,
    vị trí, góc chụp.
    """
    try:
        a = to_array(points_a, num_points * 2).reshape(num_points, 2)
        b = to_array(points_b, num_points * 2).reshape(num_points, 2)
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return 0.0
        _, _, disparity = scipy_procrustes(a, b)
        # disparity thường rất nhỏ (0 - vài chục), nhân hệ số để dễ phân biệt
        return float(1.0 / (1.0 + disparity * 10))
    except Exception:
        return 0.0


def color_list_similarity(list_a, list_b, key="rgb"):
    """
    So sánh 2 list màu (dominant_colors hoặc regional_colors), match theo
    thứ tự index, khoảng cách RGB Euclidean normalize về 0-1.
    """
    if not list_a or not list_b:
        return 0.0
    n = min(len(list_a), len(list_b))
    max_dist = np.sqrt(3) * 255
    sims = []
    for i in range(n):
        rgb_a = np.asarray(list_a[i].get(key, [0, 0, 0]), dtype=float)
        rgb_b = np.asarray(list_b[i].get(key, [0, 0, 0]), dtype=float)
        dist = np.linalg.norm(rgb_a - rgb_b)
        sims.append(1.0 - dist / max_dist)
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def categorical_similarity(a, b):
    return 1.0 if a == b else 0.0


# ============================================================
# DATASET STATS - min/max của các scalar feature trên toàn pool
# (dataset hiện có + ảnh query), dùng để chuẩn hoá tại thời điểm query
# ============================================================
class DatasetStats:
    SCALAR_FIELDS = [
        "shape_aspect_ratio", "shape_compactness", "shape_elongation", "shape_head_body_ratio",
        "texture_stripe_count", "texture_spot_count", "texture_regional_variance", "texture_scale_visibility",
        "color_brightness_mean", "color_saturation_mean", "color_variance", "color_dorsal_ventral_contrast",
    ]

    def __init__(self, rows, query_row):
        self.ranges = {}
        all_rows = list(rows) + [query_row]
        for field in self.SCALAR_FIELDS:
            values = [safe_scalar(r.get(field)) for r in all_rows]
            self.ranges[field] = (min(values), max(values))

    def sim(self, field, v1, v2):
        vmin, vmax = self.ranges[field]
        return minmax_similarity(safe_scalar(v1), safe_scalar(v2), vmin, vmax)


# ============================================================
# SIMILARITY THEO TỪNG NHÓM 10 ĐẶC TRƯNG
# (trọng số con trong mỗi nhóm cộng lại = 1.0)
# ============================================================
def shape_similarity(q, c, stats):
    scalar_sim = np.mean([
        stats.sim("shape_aspect_ratio", q["shape_aspect_ratio"], c["shape_aspect_ratio"]),
        stats.sim("shape_compactness", q["shape_compactness"], c["shape_compactness"]),
        stats.sim("shape_elongation", q["shape_elongation"], c["shape_elongation"]),
        stats.sim("shape_head_body_ratio", q["shape_head_body_ratio"], c["shape_head_body_ratio"]),
    ])
    hu_sim = euclidean_similarity(to_array(q["shape_hu_moments"], 7), to_array(c["shape_hu_moments"], 7))
    fourier_sim = euclidean_similarity(
        to_array(q["shape_fourier_descriptors"], 20), to_array(c["shape_fourier_descriptors"], 20)
    )
    fin_sim = euclidean_similarity(to_array(q["shape_fin_ratios"], 4), to_array(c["shape_fin_ratios"], 4))
    contour_sim = procrustes_similarity(q["shape_contour_points"], c["shape_contour_points"], 100)
    spine_sim = procrustes_similarity(q["shape_spine_curve"], c["shape_spine_curve"], 50)
    # NOTE: shape_landmarks bị bỏ qua - xem ghi chú đầu file (placeholder trong extractor)

    total = (
        0.25 * scalar_sim +
        0.20 * hu_sim +
        0.20 * fourier_sim +
        0.10 * fin_sim +
        0.15 * contour_sim +
        0.10 * spine_sim
    )
    return float(np.clip(total, 0.0, 1.0))


def texture_similarity(q, c, stats):
    scalar_sim = np.mean([
        stats.sim("texture_stripe_count", q["texture_stripe_count"], c["texture_stripe_count"]),
        stats.sim("texture_spot_count", q["texture_spot_count"], c["texture_spot_count"]),
        stats.sim("texture_regional_variance", q["texture_regional_variance"], c["texture_regional_variance"]),
        stats.sim("texture_scale_visibility", q["texture_scale_visibility"], c["texture_scale_visibility"]),
    ])
    glcm_sim = euclidean_similarity(to_array(q["texture_glcm"], 4), to_array(c["texture_glcm"], 4))
    lbp_sim = histogram_similarity(
        to_array(q["texture_lbp_histogram"], 256), to_array(c["texture_lbp_histogram"], 256)
    )
    gabor_sim = euclidean_similarity(to_array(q["texture_gabor_filters"], 40), to_array(c["texture_gabor_filters"], 40))
    wavelet_sim = cosine_similarity(to_array(q["texture_wavelet"], 8), to_array(c["texture_wavelet"], 8))
    haralick_sim = cosine_similarity(to_array(q["texture_haralick"], 13), to_array(c["texture_haralick"], 13))
    tamura_sim = cosine_similarity(to_array(q["texture_tamura"], 3), to_array(c["texture_tamura"], 3))

    total = (
        0.15 * scalar_sim +
        0.15 * glcm_sim +
        0.25 * lbp_sim +
        0.15 * gabor_sim +
        0.10 * wavelet_sim +
        0.10 * haralick_sim +
        0.10 * tamura_sim
    )
    return float(np.clip(total, 0.0, 1.0))


def color_similarity(q, c, stats):
    scalar_sim = np.mean([
        stats.sim("color_brightness_mean", q["color_brightness_mean"], c["color_brightness_mean"]),
        stats.sim("color_saturation_mean", q["color_saturation_mean"], c["color_saturation_mean"]),
        stats.sim("color_variance", q["color_variance"], c["color_variance"]),
        stats.sim(
            "color_dorsal_ventral_contrast", q["color_dorsal_ventral_contrast"], c["color_dorsal_ventral_contrast"]
        ),
    ])
    rgb_hist_sim = histogram_similarity(to_array(q["color_rgb_histogram"], 768), to_array(c["color_rgb_histogram"], 768))
    hsv_hist_sim = histogram_similarity(to_array(q["color_hsv_histogram"], 308), to_array(c["color_hsv_histogram"], 308))
    moments_sim = cosine_similarity(to_array(q["color_moments"], 9), to_array(c["color_moments"], 9))
    dominant_sim = color_list_similarity(q["color_dominant_colors"], c["color_dominant_colors"])
    regional_sim = color_list_similarity(q["color_regional_colors"], c["color_regional_colors"])
    hue_sim = categorical_similarity(q["color_hue_distribution_type"], c["color_hue_distribution_type"])

    total = (
        0.10 * scalar_sim +
        0.20 * rgb_hist_sim +
        0.15 * hsv_hist_sim +
        0.15 * moments_sim +
        0.20 * dominant_sim +
        0.10 * regional_sim +
        0.10 * hue_sim
    )
    return float(np.clip(total, 0.0, 1.0))


# ============================================================
# MAIN SEARCH CLASS
# ============================================================
class FishSimilaritySearch:
    """
    Tìm kiếm ảnh cá tương đồng trong database dựa trên 30 đặc trưng
    Shape + Texture + Color (relational-object database - PostgreSQL).
    """

    def __init__(self, db_config, weight_shape=0.4, weight_texture=0.3, weight_color=0.3,
                 auto_remove_bg=True, bg_model_name="u2net"):
        self.config = db_config
        self.weights = {"shape": weight_shape, "texture": weight_texture, "color": weight_color}
        self.conn = None

        # Tách nền ảnh query tự động nếu cần (xem GHI CHÚ QUAN TRỌNG VỀ ẢNH QUERY ở đầu file)
        self.auto_remove_bg = auto_remove_bg
        self.bg_model_name = bg_model_name  # phải khớp MODEL_NAME đã dùng khi tạo dataset
        self._bg_session = None  # khởi tạo 1 lần, tái sử dụng (giống background_removal.py)

    def connect(self):
        self.conn = psycopg2.connect(**self.config)
        print("✓ Đã kết nối database")

    def close(self):
        if self.conn:
            self.conn.close()
            print("✓ Đã đóng kết nối database")

    # --------------------------------------------------------
    # 0. Đảm bảo ảnh query đã tách nền GIỐNG pipeline của dataset
    # --------------------------------------------------------
    def _looks_background_removed(self, image_path, transparent_ratio_threshold=0.02):
        """
        Heuristic: ảnh được coi là "đã tách nền" nếu có kênh alpha VÀ có ít
        nhất vài % pixel gần như trong suốt (alpha ~ 0) - đặc trưng của PNG
        xuất ra từ rembg/U2Net. Ảnh JPG hoặc PNG không có vùng trong suốt
        (vd ảnh chụp thường) sẽ bị coi là CHƯA tách nền.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Không đọc được ảnh query: {image_path}")

        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            transparent_ratio = float(np.mean(alpha < 10))
            return transparent_ratio >= transparent_ratio_threshold

        return False

    def _remove_background_for_query(self, image_path):
        """
        Tách nền ảnh query bằng U2Net (rembg), tái sử dụng session giữa các
        lần gọi. Trả về đường dẫn file PNG tạm (RGBA) đã tách nền.
        """
        if not REMBG_AVAILABLE:
            raise RuntimeError(
                "Ảnh query chưa tách nền và thư viện 'rembg' chưa được cài. "
                "Cài bằng: pip install rembg  (hoặc tự tách nền ảnh query trước "
                "bằng background_removal.py rồi truyền ảnh PNG đó vào --image)."
            )

        if self._bg_session is None:
            print(f"→ Khởi tạo U2Net session ({self.bg_model_name}) để tách nền ảnh query...")
            self._bg_session = rembg_new_session(self.bg_model_name)

        image_path = Path(image_path)
        input_data = image_path.read_bytes()
        output_data = rembg_remove(input_data, session=self._bg_session)

        tmp_path = Path(tempfile.gettempdir()) / f"query_nobg_{image_path.stem}.png"
        tmp_path.write_bytes(output_data)
        print(f"  ✓ Đã tách nền ảnh query → {tmp_path}")
        return tmp_path

    def _prepare_query_image(self, image_path):
        if not self.auto_remove_bg:
            return image_path
        if self._looks_background_removed(image_path):
            print("→ Ảnh query đã có vùng trong suốt (coi như đã tách nền), dùng trực tiếp.")
            return image_path
        print("→ Ảnh query chưa tách nền, đang tách nền để khớp pipeline dataset...")
        return self._remove_background_for_query(image_path)

    # --------------------------------------------------------
    # 1. Trích xuất đặc trưng ảnh query (dùng lại 3 extractor có sẵn)
    # --------------------------------------------------------
    def extract_query_features(self, image_path):
        print(f"→ Trích xuất đặc trưng ảnh query: {image_path}")

        image_path = self._prepare_query_image(image_path)

        shape = ShapeFeatureExtractor(str(image_path)).extract_all()
        texture = TextureFeatureExtractor(str(image_path)).extract_all()
        color = ColorFeatureExtractor(str(image_path)).extract_all()

        if not (shape and texture and color):
            raise RuntimeError("Trích xuất đặc trưng ảnh query thất bại")

        return {
            "shape_aspect_ratio": shape["aspect_ratio"],
            "shape_compactness": shape["compactness"],
            "shape_elongation": shape["elongation"],
            "shape_contour_points": shape["contour_points"],
            "shape_spine_curve": shape["spine_curve"],
            "shape_fourier_descriptors": shape["fourier_descriptors"],
            "shape_hu_moments": shape["hu_moments"],
            "shape_landmarks": shape["landmarks"],
            "shape_fin_ratios": shape["fin_ratios"],
            "shape_head_body_ratio": shape["head_body_ratio"],
            "texture_glcm": texture["glcm"],
            "texture_lbp_histogram": texture["lbp_histogram"],
            "texture_gabor_filters": texture["gabor_filters"],
            "texture_stripe_count": texture["stripe_count"],
            "texture_spot_count": texture["spot_count"],
            "texture_regional_variance": texture["regional_variance"],
            "texture_scale_visibility": texture["scale_visibility"],
            "texture_wavelet": texture["wavelet"],
            "texture_haralick": texture["haralick"],
            "texture_tamura": texture["tamura"],
            "color_rgb_histogram": color["rgb_histogram"],
            "color_dominant_colors": color["dominant_colors"],
            "color_moments": color["color_moments"],
            "color_hsv_histogram": color["hsv_histogram"],
            "color_brightness_mean": color["brightness_mean"],
            "color_saturation_mean": color["saturation_mean"],
            "color_regional_colors": color["regional_colors"],
            "color_variance": color["color_variance"],
            "color_dorsal_ventral_contrast": color["dorsal_ventral_contrast"],
            "color_hue_distribution_type": color["hue_distribution_type"],
        }

    # --------------------------------------------------------
    # 2. Load toàn bộ feature_vectors từ DB kèm thông tin ảnh + loài
    # --------------------------------------------------------
    def load_candidates(self):
        sql = """
            SELECT fv.*, fi.original_path, fi.processed_path,
                   fi.species_id, fs.species_name_vn, fs.species_name_en
            FROM feature_vectors fv
            JOIN fish_images fi ON fv.image_id = fi.image_id
            JOIN fish_species fs ON fi.species_id = fs.species_id
            WHERE fi.is_feature_extracted = TRUE
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------
    # 3. Tìm kiếm Top-K ảnh tương đồng
    # --------------------------------------------------------
    def search(self, image_path, top_k=5, query_image_id=None):
        """
        Args:
            image_path: đường dẫn ảnh cá mới cần tìm kiếm
            top_k: số lượng kết quả trả về (mặc định 5)
            query_image_id: nếu ảnh query ĐÃ có image_id trong DB (vd: đang
                tính hàng xóm gần nhất cho ảnh có sẵn), truyền vào để (a) loại
                chính nó khỏi kết quả và (b) ghi kết quả vào similarity_cache.
        """
        start = time.time()

        query_features = self.extract_query_features(image_path)
        candidates = self.load_candidates()

        if query_image_id is not None:
            candidates = [c for c in candidates if c["image_id"] != query_image_id]

        if not candidates:
            print("✗ Database chưa có ảnh nào để so sánh")
            return []

        stats = DatasetStats(candidates, query_features)

        results = []
        for c in candidates:
            s_sim = shape_similarity(query_features, c, stats)
            t_sim = texture_similarity(query_features, c, stats)
            col_sim = color_similarity(query_features, c, stats)
            overall = (
                self.weights["shape"] * s_sim +
                self.weights["texture"] * t_sim +
                self.weights["color"] * col_sim
            )
            results.append({
                "image_id": c["image_id"],
                "original_path": c["original_path"],
                "processed_path": c["processed_path"],
                "species_name_vn": c["species_name_vn"],
                "species_name_en": c["species_name_en"],
                "shape_similarity": round(s_sim, 4),
                "texture_similarity": round(t_sim, 4),
                "color_similarity": round(col_sim, 4),
                "overall_similarity": round(overall, 4),
            })

        results.sort(key=lambda r: r["overall_similarity"], reverse=True)
        top_results = results[:top_k]

        elapsed_ms = int((time.time() - start) * 1000)
        self.log_query(image_path, top_k, top_results, elapsed_ms)

        if query_image_id is not None:
            self.save_to_similarity_cache(query_image_id, top_results)

        return top_results

    # --------------------------------------------------------
    # 4. Ghi log vào bảng user_queries
    # --------------------------------------------------------
    def log_query(self, image_path, top_k, results, elapsed_ms):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_queries
                    (query_image_path, top_k, weight_shape, weight_texture, weight_color,
                     result_image_ids, query_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(image_path), top_k,
                        self.weights["shape"], self.weights["texture"], self.weights["color"],
                        Json([r["image_id"] for r in results]),
                        elapsed_ms,
                    ),
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"⚠ Không lưu được log query: {e}")

    # --------------------------------------------------------
    # 5. (Tuỳ chọn) Ghi vào similarity_cache
    # --------------------------------------------------------
    def save_to_similarity_cache(self, query_image_id, results):
        try:
            with self.conn.cursor() as cur:
                for r in results:
                    cur.execute(
                        """
                        INSERT INTO similarity_cache
                        (query_image_id, similar_image_id, shape_similarity,
                         texture_similarity, color_similarity, overall_similarity,
                         weight_shape, weight_texture, weight_color)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query_image_id, r["image_id"],
                            r["shape_similarity"], r["texture_similarity"],
                            r["color_similarity"], r["overall_similarity"],
                            self.weights["shape"], self.weights["texture"], self.weights["color"],
                        ),
                    )
            self.conn.commit()
            print(f"✓ Đã lưu {len(results)} kết quả vào similarity_cache")
        except Exception as e:
            self.conn.rollback()
            print(f"⚠ Không lưu được similarity_cache: {e}")


# ============================================================
# IN KẾT QUẢ
# ============================================================
def print_results(results):
    print(f"\n{'='*80}")
    print(f"TOP {len(results)} ẢNH TƯƠNG ĐỒNG NHẤT")
    print(f"{'='*80}")
    for rank, r in enumerate(results, 1):
        print(f"\n#{rank}  [image_id={r['image_id']}]  {r['species_name_vn']} ({r['species_name_en']})")
        print(f"     Overall similarity : {r['overall_similarity']*100:.2f}%")
        print(f"     ├─ Shape   : {r['shape_similarity']*100:.2f}%")
        print(f"     ├─ Texture : {r['texture_similarity']*100:.2f}%")
        print(f"     └─ Color   : {r['color_similarity']*100:.2f}%")
        print(f"     Ảnh: {r['original_path']}")
    print()


# ============================================================
# MAIN - CLI
# ============================================================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện (đã có sẵn nếu chạy được batch_processing.py):
       pip install psycopg2-binary numpy scipy opencv-python opencv-contrib-python \
                   scikit-learn scikit-image pywavelets

    2. Chỉnh DB_CONFIG bên dưới (hoặc truyền qua biến môi trường nếu muốn)

    3. Chạy:
       python search_similar_fish.py --image "D:/test/ca_la.jpg" --top_k 5

       Tuỳ chọn trọng số (mặc định khớp similarity_cache: 0.4/0.3/0.3):
       python search_similar_fish.py --image "ca_la.jpg" --weight_shape 0.5 --weight_texture 0.3 --weight_color 0.2
    """

    parser = argparse.ArgumentParser(description="Tìm kiếm ảnh cá tương đồng trong database")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh cá cần tìm kiếm")
    parser.add_argument("--top_k", type=int, default=5, help="Số lượng kết quả trả về (mặc định 5)")
    parser.add_argument("--weight_shape", type=float, default=0.4)
    parser.add_argument("--weight_texture", type=float, default=0.3)
    parser.add_argument("--weight_color", type=float, default=0.3)
    parser.add_argument(
        "--query_image_id", type=int, default=None,
        help="Nếu ảnh query đã có image_id trong DB, truyền vào để loại khỏi kết quả và ghi similarity_cache",
    )
    parser.add_argument(
        "--bg_model", default="u2net",
        help="Model rembg dùng để tách nền ảnh query nếu cần (phải khớp model đã dùng cho dataset, mặc định u2net)",
    )
    parser.add_argument(
        "--no_auto_bg_removal", action="store_true",
        help="Tắt tự động tách nền ảnh query (dùng khi ảnh query đã chắc chắn tách nền sẵn)",
    )
    args = parser.parse_args()

    # ========== CHỈNH SỬA Ở ĐÂY (giống batch_processing.py) ==========
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fish_database',
        'user': 'postgres',
        'password': 'PassW0rd@',
    }
    # ===================================================================

    searcher = FishSimilaritySearch(
        DB_CONFIG,
        weight_shape=args.weight_shape,
        weight_texture=args.weight_texture,
        weight_color=args.weight_color,
        auto_remove_bg=not args.no_auto_bg_removal,
        bg_model_name=args.bg_model,
    )
    searcher.connect()
    try:
        results = searcher.search(args.image, top_k=args.top_k, query_image_id=args.query_image_id)
        print_results(results)
    finally:
        searcher.close()