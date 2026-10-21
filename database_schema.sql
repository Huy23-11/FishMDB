-- =====================================================
-- FISH IMAGE DATABASE SCHEMA
-- Relational-Object Database Design
-- =====================================================

-- ========================================
-- TABLE 1: fish_species (Bảng thông tin loài cá)
-- ========================================
CREATE TABLE fish_species (
    species_id SERIAL PRIMARY KEY,
    species_name_vn VARCHAR(100) NOT NULL,        -- Tên tiếng Việt (vd: Cá Lóc)
    species_name_en VARCHAR(100),                  -- Tên tiếng Anh (vd: Snakehead)
    scientific_name VARCHAR(150),                  -- Tên khoa học (vd: Channa striata)
    description TEXT,                              -- Mô tả loài
    habitat VARCHAR(50),                           -- Môi trường sống: freshwater, saltwater, brackish
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(species_name_en)
);

-- Index cho tìm kiếm nhanh
CREATE INDEX idx_species_name ON fish_species(species_name_en);


-- ========================================
-- TABLE 2: fish_images (Bảng ảnh cá)
-- ========================================
CREATE TABLE fish_images (
    image_id SERIAL PRIMARY KEY,
    species_id INTEGER NOT NULL REFERENCES fish_species(species_id) ON DELETE CASCADE,

    -- Đường dẫn file
    original_path VARCHAR(500) NOT NULL,           -- Đường dẫn ảnh JPG gốc
    processed_path VARCHAR(500),                   -- Đường dẫn ảnh PNG đã tách nền

    -- Metadata ảnh
    width INTEGER,                                 -- Chiều rộng (pixels)
    height INTEGER,                                -- Chiều cao (pixels)
    file_size_kb INTEGER,                          -- Kích thước file (KB)
    format VARCHAR(10),                            -- jpg, png

    -- Thông tin thu thập
    source VARCHAR(100),                           -- Nguồn ảnh (web, camera, dataset)
    capture_date DATE,                             -- Ngày chụp
    notes TEXT,                                    -- Ghi chú

    -- Trạng thái xử lý
    is_background_removed BOOLEAN DEFAULT FALSE,   -- Đã tách nền chưa
    is_feature_extracted BOOLEAN DEFAULT FALSE,    -- Đã trích xuất đặc trưng chưa

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(original_path)
);

-- Index cho query nhanh
CREATE INDEX idx_image_species ON fish_images(species_id);
CREATE INDEX idx_image_processed ON fish_images(is_feature_extracted);


-- ========================================
-- TABLE 3: feature_vectors (Bảng vector đặc trưng)
-- ========================================
-- MỖI ẢNH có 1 row duy nhất chứa vector 30 chiều

CREATE TABLE feature_vectors (
    vector_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES fish_images(image_id) ON DELETE CASCADE,

    -- ==================
    -- SHAPE FEATURES (10)
    -- ==================
    shape_aspect_ratio REAL,                      -- 1. Aspect Ratio
    shape_compactness REAL,                       -- 2. Compactness
    shape_elongation REAL,                        -- 3. Elongation
    shape_contour_points JSON,                    -- 4. Contour (100 points) - lưu dạng JSON array
    shape_spine_curve JSON,                       -- 5. Spine Curve (50 points) - JSON array
    shape_fourier_descriptors JSON,               -- 6. Fourier Descriptors (20 coefficients) - JSON array
    shape_hu_moments JSON,                        -- 7. Hu Moments (7 values) - JSON array
    shape_landmarks JSON,                         -- 8. Landmarks (10 points) - JSON array
    shape_fin_ratios JSON,                        -- 9. Fin Ratios (4 values) - JSON array
    shape_head_body_ratio REAL,                   -- 10. Head/Body Ratio

    -- ==================
    -- TEXTURE FEATURES (10)
    -- ==================
    texture_glcm JSON,                            -- 11. GLCM (4 values: contrast, homogeneity, correlation, energy)
    texture_lbp_histogram JSON,                   -- 12. LBP Histogram (256 bins) - JSON array
    texture_gabor_filters JSON,                   -- 13. Gabor Filters (40 values: 5 scales × 8 orientations)
    texture_stripe_count INTEGER,                 -- 14. Stripe Detection (số lượng sọc)
    texture_spot_count INTEGER,                   -- 15. Spot Detection (số lượng đốm)
    texture_regional_variance REAL,               -- 16. Regional Texture Variance
    texture_scale_visibility REAL,                -- 17. Scale Visibility Score
    texture_wavelet JSON,                         -- 18. Wavelet Coefficients (JSON array)
    texture_haralick JSON,                        -- 19. Haralick Features (13 values)
    texture_tamura JSON,                          -- 20. Tamura Features (3 values: coarseness, contrast, directionality)

    -- ==================
    -- COLOR FEATURES (10)
    -- ==================
    color_rgb_histogram JSON,                     -- 21. RGB Histogram (768 bins: 256*3) - JSON array
    color_dominant_colors JSON,                   -- 22. K-means Dominant Colors (top 3 colors with %) - JSON array
    color_moments JSON,                           -- 23. Color Moments (9 values: mean/std/skew for R,G,B)
    color_hsv_histogram JSON,                     -- 24. HSV Histogram (308 bins) - JSON array
    color_brightness_mean REAL,                   -- 25. Brightness Mean
    color_saturation_mean REAL,                   -- 26. Saturation Mean
    color_regional_colors JSON,                   -- 27. Regional Colors (4 zones) - JSON array
    color_variance REAL,                          -- 28. Color Variance
    color_dorsal_ventral_contrast REAL,           -- 29. Dorsal-Ventral Contrast (countershading)
    color_hue_distribution_type VARCHAR(20),      -- 30. Hue Distribution Type (uniform/dominant/bimodal)

    -- Metadata
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extraction_version VARCHAR(20),                -- Version của thuật toán trích xuất (vd: "v1.0")

    UNIQUE(image_id)
);

-- Index cho tìm kiếm nhanh vector của 1 ảnh
CREATE INDEX idx_vector_image ON feature_vectors(image_id);


-- ========================================
-- TABLE 4: similarity_cache (Cache kết quả tìm kiếm)
-- ========================================
-- Lưu kết quả similarity search để tăng tốc query lặp lại

CREATE TABLE similarity_cache (
    cache_id SERIAL PRIMARY KEY,
    query_image_id INTEGER NOT NULL REFERENCES fish_images(image_id) ON DELETE CASCADE,
    similar_image_id INTEGER NOT NULL REFERENCES fish_images(image_id) ON DELETE CASCADE,

    -- Similarity scores
    shape_similarity REAL,                        -- Độ tương đồng Shape (0-1)
    texture_similarity REAL,                      -- Độ tương đồng Texture (0-1)
    color_similarity REAL,                        -- Độ tương đồng Color (0-1)
    overall_similarity REAL,                      -- Tổng hợp (weighted average)

    -- Weights sử dụng
    weight_shape REAL DEFAULT 0.4,                -- Trọng số Shape
    weight_texture REAL DEFAULT 0.3,              -- Trọng số Texture
    weight_color REAL DEFAULT 0.3,                -- Trọng số Color

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index cho query top-K similar images
CREATE INDEX idx_cache_query ON similarity_cache(query_image_id, overall_similarity DESC);
CREATE INDEX idx_cache_similar ON similarity_cache(similar_image_id);


-- ========================================
-- TABLE 5: user_queries (Lịch sử tìm kiếm)
-- ========================================
CREATE TABLE user_queries (
    query_id SERIAL PRIMARY KEY,
    query_image_path VARCHAR(500),                -- Đường dẫn ảnh query
    query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Tham số query
    top_k INTEGER DEFAULT 5,                      -- Số lượng kết quả trả về
    weight_shape REAL,
    weight_texture REAL,
    weight_color REAL,

    -- Kết quả (JSON array chứa image_ids)
    result_image_ids JSON,                        -- [123, 456, 789, 101, 202]

    -- Performance
    query_time_ms INTEGER,                        -- Thời gian xử lý (milliseconds)

    notes TEXT
);

CREATE INDEX idx_query_timestamp ON user_queries(query_timestamp DESC);


-- ========================================
-- VIEWS - Truy vấn thông tin tổng hợp
-- ========================================

-- View: Thống kê số lượng ảnh theo loài
CREATE VIEW species_image_count AS
SELECT
    s.species_id,
    s.species_name_vn,
    s.species_name_en,
    COUNT(i.image_id) AS image_count,
    SUM(CASE WHEN i.is_feature_extracted THEN 1 ELSE 0 END) AS processed_count
FROM fish_species s
LEFT JOIN fish_images i ON s.species_id = i.species_id
GROUP BY s.species_id, s.species_name_vn, s.species_name_en
ORDER BY image_count DESC;


-- View: Ảnh chưa trích xuất đặc trưng
CREATE VIEW images_pending_extraction AS
SELECT
    i.image_id,
    s.species_name_en,
    i.original_path,
    i.processed_path,
    i.is_background_removed
FROM fish_images i
JOIN fish_species s ON i.species_id = s.species_id
WHERE i.is_feature_extracted = FALSE
ORDER BY i.created_at;


-- ========================================
-- STORED FUNCTIONS - Các hàm tiện ích
-- ========================================

-- Hàm tính Euclidean Distance giữa 2 vector
-- (Sẽ dùng để tính similarity trong application layer)

-- Trigger: Tự động update timestamp khi sửa fish_images
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_fish_images_updated_at
BEFORE UPDATE ON fish_images
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


-- ========================================
-- SAMPLE DATA - Dữ liệu mẫu
-- ========================================

-- Insert một số loài cá phổ biến
INSERT INTO fish_species (species_name_vn, species_name_en, scientific_name, habitat) VALUES
('Cá Lóc', 'Snakehead', 'Channa striata', 'freshwater'),
('Cá Dao', 'Knifefish', 'Notopterus notopterus', 'freshwater'),
('Cá Lươn', 'Freshwater Eel', 'Monopterus albus', 'freshwater'),
('Cá Bơn', 'Flounder', 'Paralichthys olivaceus', 'saltwater'),
('Cá Mặt Trăng', 'Mola Mola', 'Mola mola', 'saltwater'),
('Cá Sư Tử', 'Lionfish', 'Pterois volitans', 'saltwater'),
('Cá Ngựa', 'Seahorse', 'Hippocampus kuda', 'saltwater'),
('Cá Vàng', 'Goldfish', 'Carassius auratus', 'freshwater'),
('Cá Hề', 'Clownfish', 'Amphiprion ocellaris', 'saltwater'),
('Cá Hồi', 'Salmon', 'Salmo salar', 'freshwater');


-- ========================================
-- NOTES - Lưu ý quan trọng
-- ========================================

/*
1. DATABASE ENGINE: Khuyên dùng PostgreSQL
   - Hỗ trợ JSON type native
   - Performance tốt với GiST index cho vector search
   - Có extension pgvector để tối ưu similarity search

2. JSON FIELDS: Lưu arrays dạng JSON
   - shape_contour_points: [x1, y1, x2, y2, ..., x100, y100]
   - texture_lbp_histogram: [v1, v2, ..., v256]
   - color_rgb_histogram: [r1, r2, ..., r256, g1, ..., g256, b1, ..., b256]

3. SIMILARITY SEARCH:
   - Trong application layer (Python), load vectors ra rồi tính distance
   - Hoặc dùng PostgreSQL extension pgvector để tính trong DB
   - Cache kết quả trong similarity_cache để tăng tốc

4. STORAGE:
   - Ảnh gốc và processed lưu trên filesystem
   - Database chỉ lưu đường dẫn (paths) và metadata
   - Vector đặc trưng lưu trong DB

5. SCALING:
   - Với >10,000 ảnh, cân nhắc dùng pgvector
   - Với >100,000 ảnh, cân nhắc Approximate Nearest Neighbor (ANN) search
   - (FAISS, Annoy, HNSW)
*/
