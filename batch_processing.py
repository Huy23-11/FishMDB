"""
BATCH PROCESSING - Xử lý toàn bộ dataset và lưu vào database
Kết hợp 3 extractors: Shape, Texture, Color

Flow:
1. Quét thư mục ảnh PNG đã tách nền
2. Trích xuất 30 đặc trưng (Shape + Texture + Color)
3. Lưu vào PostgreSQL database
4. Log tiến trình và lỗi
"""

import os
import json
import numpy as np
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from datetime import datetime
import time

# Import 3 extractors
from extract_shape_features import ShapeFeatureExtractor
from extract_texture_features import TextureFeatureExtractor
from extract_color_features import ColorFeatureExtractor


def to_native(obj):
    """
    Convert đệ quy các kiểu numpy (int32, float32, ndarray...) sang kiểu
    Python thuần (int, float, list), vì json.dumps (dùng bên trong
    psycopg2.extras.Json) không serialize được kiểu numpy.
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # numpy scalar: int32, float32, bool_, ...
        return obj.item()
    else:
        return obj


class DatabaseManager:
    """
    Quản lý kết nối và operations với PostgreSQL database
    """

    def __init__(self, db_config):
        """
        Args:
            db_config: dict chứa {host, port, database, user, password}
        """
        self.config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """Kết nối database"""
        try:
            self.conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.cursor = self.conn.cursor()
            print("✓ Đã kết nối database")
            return True

        except Exception as e:
            print(f"✗ Lỗi kết nối database: {str(e)}")
            return False

    def close(self):
        """Đóng kết nối"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Đã đóng kết nối database")

    def insert_species(self, species_name_vn, species_name_en, scientific_name=None, habitat=None):
        """
        Thêm loài cá mới vào bảng fish_species.
        Nếu đã tồn tại, trả về species_id.
        """
        try:
            # Check xem species đã tồn tại chưa
            self.cursor.execute(
                "SELECT species_id FROM fish_species WHERE species_name_en = %s",
                (species_name_en,)
            )
            result = self.cursor.fetchone()

            if result:
                return result[0]  # species_id

            # Insert species mới
            self.cursor.execute(
                """
                INSERT INTO fish_species (species_name_vn, species_name_en, scientific_name, habitat)
                VALUES (%s, %s, %s, %s)
                RETURNING species_id
                """,
                (species_name_vn, species_name_en, scientific_name, habitat)
            )
            species_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return species_id

        except Exception as e:
            self.conn.rollback()
            print(f"✗ Lỗi insert species: {str(e)}")
            return None

    def get_existing_image(self, original_path):
        """
        Kiểm tra xem ảnh (theo original_path) đã có trong DB chưa.

        Returns:
            tuple (image_id, is_feature_extracted) nếu đã tồn tại, ngược lại None
        """
        try:
            self.cursor.execute(
                "SELECT image_id, is_feature_extracted FROM fish_images WHERE original_path = %s",
                (original_path,)
            )
            result = self.cursor.fetchone()
            if result:
                return (result[0], result[1])
            return None

        except Exception as e:
            print(f"✗ Lỗi kiểm tra image tồn tại: {str(e)}")
            return None

    def insert_image(self, species_id, original_path, processed_path, width, height, file_size_kb):
        """
        Thêm ảnh mới vào bảng fish_images.
        Nếu original_path đã tồn tại (ảnh đã được xử lý ở lần chạy trước),
        trả về image_id có sẵn thay vì insert mới (tránh lỗi duplicate key).
        """
        # Check đã tồn tại chưa trước (resume logic)
        existing = self.get_existing_image(original_path)
        if existing:
            return existing[0]  # trả về image_id có sẵn

        try:
            self.cursor.execute(
                """
                INSERT INTO fish_images
                (species_id, original_path, processed_path, width, height, file_size_kb,
                 format, is_background_removed, is_feature_extracted)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING image_id
                """,
                (species_id, original_path, processed_path, width, height, file_size_kb,
                 'png', True, False)
            )
            image_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return image_id

        except Exception as e:
            self.conn.rollback()
            print(f"✗ Lỗi insert image: {str(e)}")
            return None

    def insert_features(self, image_id, shape_features, texture_features, color_features):
        """
        Lưu vector đặc trưng vào bảng feature_vectors.
        """
        try:
            self.cursor.execute(
                """
                INSERT INTO feature_vectors (
                    image_id,
                    shape_aspect_ratio,
                    shape_compactness,
                    shape_elongation,
                    shape_contour_points,
                    shape_spine_curve,
                    shape_fourier_descriptors,
                    shape_hu_moments,
                    shape_landmarks,
                    shape_fin_ratios,
                    shape_head_body_ratio,
                    texture_glcm,
                    texture_lbp_histogram,
                    texture_gabor_filters,
                    texture_stripe_count,
                    texture_spot_count,
                    texture_regional_variance,
                    texture_scale_visibility,
                    texture_wavelet,
                    texture_haralick,
                    texture_tamura,
                    color_rgb_histogram,
                    color_dominant_colors,
                    color_moments,
                    color_hsv_histogram,
                    color_brightness_mean,
                    color_saturation_mean,
                    color_regional_colors,
                    color_variance,
                    color_dorsal_ventral_contrast,
                    color_hue_distribution_type,
                    extraction_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    image_id,
                    # Shape (10)
                    shape_features['aspect_ratio'],
                    shape_features['compactness'],
                    shape_features['elongation'],
                    Json(shape_features['contour_points']),
                    Json(shape_features['spine_curve']),
                    Json(shape_features['fourier_descriptors']),
                    Json(shape_features['hu_moments']),
                    Json(shape_features['landmarks']),
                    Json(shape_features['fin_ratios']),
                    shape_features['head_body_ratio'],
                    # Texture (10)
                    Json(texture_features['glcm']),
                    Json(texture_features['lbp_histogram']),
                    Json(texture_features['gabor_filters']),
                    texture_features['stripe_count'],
                    texture_features['spot_count'],
                    texture_features['regional_variance'],
                    texture_features['scale_visibility'],
                    Json(texture_features['wavelet']),
                    Json(texture_features['haralick']),
                    Json(texture_features['tamura']),
                    # Color (10)
                    Json(color_features['rgb_histogram']),
                    Json(color_features['dominant_colors']),
                    Json(color_features['color_moments']),
                    Json(color_features['hsv_histogram']),
                    color_features['brightness_mean'],
                    color_features['saturation_mean'],
                    Json(color_features['regional_colors']),
                    color_features['color_variance'],
                    color_features['dorsal_ventral_contrast'],
                    color_features['hue_distribution_type'],
                    # Version
                    'v1.0'
                )
            )

            # Update flag is_feature_extracted
            self.cursor.execute(
                "UPDATE fish_images SET is_feature_extracted = TRUE WHERE image_id = %s",
                (image_id,)
            )

            self.conn.commit()
            return True

        except Exception as e:
            self.conn.rollback()
            print(f"✗ Lỗi insert features: {str(e)}")
            return False


class BatchProcessor:
    """
    Xử lý batch toàn bộ thư mục ảnh
    """

    def __init__(self, db_manager, input_folder, species_mapping):
        """
        Args:
            db_manager: DatabaseManager instance
            input_folder: Thư mục chứa ảnh PNG đã tách nền
            species_mapping: dict {filename_pattern: (species_name_vn, species_name_en)}
        """
        self.db = db_manager
        self.input_folder = Path(input_folder)
        self.species_mapping = species_mapping

        # Stats
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        # Error log
        self.errors = []

    def parse_species_from_filename(self, filename):
        """
        Parse species từ tên file.

        Ví dụ: "ca_loc_001_nobg.png" → ("Cá Lóc", "Snakehead")
        """
        for pattern, (name_vn, name_en) in self.species_mapping.items():
            if pattern.lower() in filename.lower():
                return (name_vn, name_en)

        # Default nếu không match
        return ("Unknown", "Unknown")

    def process_single_image(self, image_path):
        """
        Xử lý 1 ảnh: trích xuất features và lưu vào database.

        Returns:
            tuple: (success: bool, error_msg: str)
        """
        try:
            filename = image_path.name
            print(f"\n[{self.stats['success'] + self.stats['failed'] + 1}/{self.stats['total']}] Xử lý: {filename}")

            # 0. Kiểm tra đã xử lý xong (resume) chưa - skip nếu đã có feature_vectors
            existing = self.db.get_existing_image(str(image_path))
            if existing and existing[1]:  # is_feature_extracted == True
                print("  ⊘ Đã xử lý trước đó, bỏ qua.")
                return (True, "SKIPPED")

            # 1. Parse species
            species_vn, species_en = self.parse_species_from_filename(filename)

            # 2. Insert/get species_id
            species_id = self.db.insert_species(species_vn, species_en)
            if not species_id:
                return (False, "Không thể insert species")

            # 3. Lấy thông tin ảnh
            import cv2
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return (False, "Không đọc được ảnh")

            height, width = img.shape[:2]
            file_size_kb = image_path.stat().st_size // 1024

            # 4. Insert image
            image_id = self.db.insert_image(
                species_id,
                str(image_path),  # original_path (tạm để path hiện tại)
                str(image_path),  # processed_path
                width, height, file_size_kb
            )
            if not image_id:
                return (False, "Không thể insert image")

            # 5. Trích xuất Shape features
            print("  → Trích xuất Shape...", end=" ")
            shape_extractor = ShapeFeatureExtractor(str(image_path))
            shape_features = shape_extractor.extract_all()
            if not shape_features:
                return (False, "Lỗi trích xuất Shape")
            print("✓")

            # 6. Trích xuất Texture features
            print("  → Trích xuất Texture...", end=" ")
            texture_extractor = TextureFeatureExtractor(str(image_path))
            texture_features = texture_extractor.extract_all()
            if not texture_features:
                return (False, "Lỗi trích xuất Texture")
            print("✓")

            # 7. Trích xuất Color features
            print("  → Trích xuất Color...", end=" ")
            color_extractor = ColorFeatureExtractor(str(image_path))
            color_features = color_extractor.extract_all()
            if not color_features:
                return (False, "Lỗi trích xuất Color")
            print("✓")

            # 8. Lưu features vào database
            print("  → Lưu vào database...", end=" ")
            # Convert numpy types (int32, float32, ndarray...) sang Python thuần
            # trước khi JSON-serialize, tránh lỗi "Object of type int32 is not JSON serializable"
            shape_features = to_native(shape_features)
            texture_features = to_native(texture_features)
            color_features = to_native(color_features)
            success = self.db.insert_features(image_id, shape_features, texture_features, color_features)
            if not success:
                return (False, "Lỗi lưu features vào database")
            print("✓")

            return (True, None)

        except Exception as e:
            return (False, str(e))

    def process_all(self):
        """
        Xử lý toàn bộ ảnh trong thư mục.
        """
        # Lấy danh sách ảnh PNG
        # NOTE: trên Windows (filesystem không phân biệt hoa/thường), glob("*.png")
        # và glob("*.PNG") trả về CÙNG một tập file -> phải dedupe bằng set(),
        # nếu không mỗi ảnh sẽ bị đếm và xử lý 2 lần.
        image_files = sorted(set(self.input_folder.glob("*.png")) | set(self.input_folder.glob("*.PNG")))
        self.stats['total'] = len(image_files)

        print(f"\n{'='*70}")
        print(f"Bắt đầu batch processing")
        print(f"Thư mục: {self.input_folder}")
        print(f"Tổng số ảnh: {self.stats['total']}")
        print(f"{'='*70}")

        start_time = time.time()

        for image_path in image_files:
            success, error_msg = self.process_single_image(image_path)

            if success and error_msg == "SKIPPED":
                self.stats['skipped'] += 1
            elif success:
                self.stats['success'] += 1
                print(f"  ✓ Thành công!")
            else:
                self.stats['failed'] += 1
                print(f"  ✗ LỖI: {error_msg}")
                self.errors.append({
                    'file': image_path.name,
                    'error': error_msg
                })

        elapsed_time = time.time() - start_time

        # Print summary
        print(f"\n{'='*70}")
        print(f"HOÀN THÀNH!")
        print(f"{'='*70}")
        print(f"Tổng số ảnh:       {self.stats['total']}")
        print(f"✓ Thành công:      {self.stats['success']}")
        print(f"✗ Thất bại:        {self.stats['failed']}")
        print(f"⊘ Bỏ qua:          {self.stats['skipped']}")
        print(f"Thời gian:         {elapsed_time:.2f}s ({elapsed_time/self.stats['total']:.2f}s/ảnh)")
        print(f"{'='*70}")

        # Print errors
        if self.errors:
            print(f"\nCHI TIẾT LỖI:")
            for err in self.errors:
                print(f"  - {err['file']}: {err['error']}")

        # Save log
        log_file = self.input_folder / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'errors': self.errors,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed_time
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Log đã lưu: {log_file}")


# ====================================
# MAIN - Chạy batch processing
# ====================================
if __name__ == "__main__":
    """
    HƯỚNG DẪN SỬ DỤNG:

    1. Cài đặt thư viện:
       pip install psycopg2-binary

    2. Cấu hình database và thư mục:
       Chỉnh sửa DB_CONFIG và INPUT_FOLDER bên dưới

    3. Cấu hình species mapping:
       Map filename patterns với species names

    4. Chạy:
       python batch_processing.py
    """

    # ========== CHỈNH SỬA Ở ĐÂY ==========

    # Database config
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fish_database',
        'user': 'postgres',
        'password': 'PassW0rd@'
    }

    # Input folder (ảnh PNG đã tách nền)
    INPUT_FOLDER = r"C:\Users\messi2015\FishMDB\FishDataset\processed_u2net"

    # Species mapping (filename pattern → (name_vn, name_en))
    SPECIES_MAPPING = {
        # 1. Nhóm cá trong bộ dữ liệu gốc (Kaggle)
        'bangus': ('Cá Mòng Biển', 'Bangus'),
        'big_head_carp': ('Cá Mè Hoa', 'Big Head Carp'),
        'black_spotted_barb': ('Cá Mè Vinh Đốm Đen', 'Black Spotted Barb'),
        'catfish': ('Cá Trê', 'Catfish'),
        'climbing_perch': ('Cá Rô Đồng', 'Climbing Perch'),
        'fourfinger_threadfin': ('Cá Chét', 'Fourfinger Threadfin'),
        'ca_luon': ('Cá Lươn', 'Freshwater Eel'),
        'glass_perchlet': ('Cá Sơn Thủy Tinh', 'Glass Perchlet'),
        'goby': ('Cá Bống', 'Goby'),
        'ca_vang': ('Cá Vàng', 'Goldfish'),
        'gourami': ('Cá Sặc', 'Gourami'),
        'grass_carp': ('Cá Trắm Cỏ', 'Grass Carp'),
        'green_spotted_puffer': ('Cá Nóc Chấm Xanh', 'Green Spotted Puffer'),
        'indian_carp': ('Cá Chép Ấn Độ', 'Indian Carp'),
        'indo_pacific_tarpon': ('Cá Chép Biển Ấn - Thái', 'Indo-Pacific Tarpon'),
        'jaguar_gapote': ('Cá Cichlid Jaguar', 'Jaguar Gapote'),
        'janitor_fish': ('Cá Lau Kính', 'Janitor Fish'),
        'ca_dao': ('Cá Dao', 'Knifefish'),
        'long_snouted_pipefish': ('Cá Chìa Vôi Mõm Dài', 'Long-Snouted Pipefish'),
        'mosquito_fish': ('Cá Muỗi', 'Mosquito Fish'),
        'ca_loc': ('Cá Lóc', 'Snakehead'),
        'mullet': ('Cá Đối', 'Mullet'),
        'pangasius': ('Cá Tra', 'Pangasius'),
        'perch': ('Cá Vược Châu Âu', 'Perch'),
        'scat_fish': ('Cá Đối Mục', 'Scat Fish'),
        'silver_barb': ('Cá Mè Vinh', 'Silver Barb'),
        'silver_carp': ('Cá Mè Trắng', 'Silver Carp'),
        'silver_perch': ('Cá Rô Bạc', 'Silver Perch'),
        'tenpounder': ('Cá Rựa', 'Tenpounder'),
        'tilapia': ('Cá Rô Phi', 'Tilapia'),

        # 2. Nhóm cá nước ngọt, nước lợ và đặc sản sông ngòi bổ sung
        'ca_hoi': ('Cá Hồi', 'Salmon'),
        'bass': ('Cá Vược', 'Bass'),
        'sturgeon': ('Cá Tầm', 'Sturgeon'),
        'common_carp': ('Cá Chép Thường', 'Common Carp'),
        'red_tilapia': ('Cá Diêu Hồng', 'Red Tilapia'),
        'black_carp': ('Cá Trắm Đen', 'Black Carp'),
        'loach': ('Cá Chạch', 'Loach'),
        'bronze_featherback': ('Cá Thát Lát', 'Bronze Featherback'),
        'snakeskin_gourami': ('Cá Sặc Rằn', 'Snakeskin Gourami'),
        'mud_carp': ('Cá Trôi', 'Mud Carp'),
        'mudskipper': ('Cá Kèo', 'Mudskipper'),
        'bagrid_catfish': ('Cá Lăng', 'Bagrid Catfish'),
        'barramundi': ('Cá Chẽm', 'Barramundi'),
        'giant_barb': ('Cá Hô', 'Giant Barb'),
        'ca_linh': ('Cá Linh', 'Siamese Mud Carp'),
        'alligator_gar': ('Cá Sấu Hỏa Tiễn', 'Alligator Gar'),
        'arapaima': ('Cá Hải Tượng', 'Arapaima'),
        'giant_gourami': ('Cá Tai Tượng', 'Giant Gourami'),
        'bong_tuong': ('Cá Bống Tượng', 'Marble Goby'),
        'ca_ngat': ('Cá Ngát', 'Eeltail Catfish'),
        'ca_dua': ('Cá Dứa', 'Tra Catfish'),
        'leather_carp': ('Cá Chép Không Vảy', 'Leather Carp'),

        # 3. Nhóm cá biển và thương mại bổ sung
        'ca_ngu': ('Cá Ngừ', 'Tuna'),
        'mackerel': ('Cá Thu', 'Mackerel'),
        'swordfish': ('Cá Kiếm', 'Swordfish'),
        'grouper': ('Cá Mú', 'Grouper'),
        'ca_bon': ('Cá Bơn', 'Flounder'),
        'red_snapper': ('Cá Hồng', 'Red Snapper'),
        'stingray': ('Cá Đuối', 'Stingray'),
        'ca_he': ('Cá Hề', 'Clownfish'),
        'scad': ('Cá Nục', 'Scad'),
        'pomfret': ('Cá Chim', 'Pomfret'),
        'yellowstripe_scad': ('Cá Chỉ Vàng', 'Yellowstripe Scad'),
        'sardine': ('Cá Trích', 'Sardine'),
        'triggerfish': ('Cá Bò', 'Triggerfish'),
        'cobia': ('Cá Bớp', 'Cobia'),
        'anchovy': ('Cá Cơm', 'Anchovy'),
        'bombay_duck': ('Cá Khoai', 'Bombay Duck'),
        'shark': ('Cá Mập', 'Shark'),
        'moray_eel': ('Cá Chình', 'Moray Eel'),

        # 4. Nhóm cá cảnh phổ biến
        'betta': ('Cá Xiêm', 'Betta'),
        'guppy': ('Cá Bảy Màu', 'Guppy'),
        'flowerhorn': ('Cá La Hán', 'Flowerhorn'),
        'arowana': ('Cá Rồng', 'Arowana'),
        'discus': ('Cá Dĩa', 'Discus'),
        'koi': ('Cá Koi', 'Koi'),
        'angelfish': ('Cá Thần Tiên', 'Angelfish'),
        'neon_tetra': ('Cá Neon', 'Neon Tetra'),
        'swordtail': ('Cá Đuôi Kiếm', 'Swordtail'),
        'corydoras': ('Cá Chuột', 'Corydoras'),
        'clown_loach': ('Cá Heo Nước Ngọt', 'Clown Loach'),
        'blood_parrot': ('Cá Hồng Két', 'Blood Parrot Cichlid'),
        'bala_shark': ('Cá Mập Thái', 'Bala Shark'),
        'ram_cichlid': ('Cá Phượng Hoàng', 'Ram Cichlid'),

        # 5. Nhóm cá có hình dáng và đặc điểm kỳ dị, độc lạ
        'seahorse': ('Cá Ngựa', 'Seahorse'),
        'ca_su_tu': ('Cá Sư Tử', 'Lionfish'),
        'ca_mola': ('Cá Mặt Trăng', 'Mola Mola'),
        'flying_fish': ('Cá Chuồn', 'Flying Fish'),
        'blue_tang': ('Cá Đuôi Gai', 'Blue Tang'),
        'butterflyfish': ('Cá Bướm', 'Butterflyfish'),
        'parrotfish': ('Cá Mó', 'Parrotfish'),
        'emperor_angelfish': ('Cá Hoàng Đế', 'Emperor Angelfish'),
        'damselfish': ('Cá Thia Biển', 'Damselfish'),
        'porcupinefish': ('Cá Nóc Nhím', 'Porcupinefish'),
        'hairtail': ('Cá Hố', 'Largehead Hairtail'),
        'trumpetfish': ('Cá Chìa Vôi Biển', 'Trumpetfish'),
        'frogfish': ('Cá Mao Ếch', 'Frogfish'),
        'batfish': ('Cá Dơi', 'Batfish'),
        'napoleon_wrasse': ('Cá Bàng Chài Napoleon', 'Napoleon Wrasse'),
        'mandarinfish': ('Cá Đàn Lia Gai', 'Mandarinfish'),
        'pleco': ('Cá Tỳ Bà', 'Pleco'),
        'lanternfish': ('Cá Lồng Đèn', 'Lanternfish'),
        'oarfish': ('Cá Mái Chèo', 'Oarfish'),
        'stonefish': ('Cá Mặt Quỷ', 'Stonefish'),
        'archerfish': ('Cá Mang Rổ', 'Archerfish'),
        'coelacanth': ('Cá Vây Tay', 'Coelacanth'),
        'moorish_idol': ('Cá Thù Lù', 'Moorish Idol'),
        'elephantnose_fish': ('Cá Nheo Mũi Hình Ống', 'Elephantnose Fish'),
    }

    # =====================================

    # Khởi tạo database manager
    db_manager = DatabaseManager(DB_CONFIG)

    if not db_manager.connect():
        print("✗ Không thể kết nối database. Kiểm tra lại cấu hình.")
        exit(1)

    try:
        # Khởi tạo batch processor
        processor = BatchProcessor(db_manager, INPUT_FOLDER, SPECIES_MAPPING)

        # Chạy batch processing
        processor.process_all()

    finally:
        # Đóng kết nối
        db_manager.close()

    print("\n✓ Done!")