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
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from datetime import datetime
import time

# Import 3 extractors
from extract_shape_features import ShapeFeatureExtractor
from extract_texture_features import TextureFeatureExtractor
from extract_color_features import ColorFeatureExtractor


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

    def insert_image(self, species_id, original_path, processed_path, width, height, file_size_kb):
        """
        Thêm ảnh mới vào bảng fish_images.
        """
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
        image_files = list(self.input_folder.glob("*.png")) + list(self.input_folder.glob("*.PNG"))
        self.stats['total'] = len(image_files)

        print(f"\n{'='*70}")
        print(f"Bắt đầu batch processing")
        print(f"Thư mục: {self.input_folder}")
        print(f"Tổng số ảnh: {self.stats['total']}")
        print(f"{'='*70}")

        start_time = time.time()

        for image_path in image_files:
            success, error_msg = self.process_single_image(image_path)

            if success:
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
        'password': 'your_password'
    }

    # Input folder (ảnh PNG đã tách nền)
    INPUT_FOLDER = "D:/FishDataset/processed"

    # Species mapping (filename pattern → (name_vn, name_en))
    SPECIES_MAPPING = {
        'ca_loc': ('Cá Lóc', 'Snakehead'),
        'ca_dao': ('Cá Dao', 'Knifefish'),
        'ca_luon': ('Cá Lươn', 'Freshwater Eel'),
        'ca_bon': ('Cá Bơn', 'Flounder'),
        'ca_mola': ('Cá Mặt Trăng', 'Mola Mola'),
        'ca_su_tu': ('Cá Sư Tử', 'Lionfish'),
        'ca_vang': ('Cá Vàng', 'Goldfish'),
        'ca_he': ('Cá Hề', 'Clownfish'),
        'ca_hoi': ('Cá Hồi', 'Salmon'),
        'ca_ngu': ('Cá Ngừ', 'Tuna'),
        'koi': ('Cá Koi', 'Koi'),
        'betta': ('Cá Xiêm', 'Betta'),
        # Thêm các loài khác...
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
