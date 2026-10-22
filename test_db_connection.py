import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fish_database',
    'user': 'postgres',
    'password': 'PassW0rd@'  # ← ĐỔI MẬT KHẨU Ở ĐÂY
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✓ Kết nối database thành công!")
    conn.close()
except Exception as e:
    print(f"✗ Lỗi: {str(e)}")