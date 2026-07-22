import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

# ----------------------------
# 1️⃣ Database Connection
# ----------------------------
username = 'root'
password = quote_plus('Teefa18@')  # Encode special characters in password
host = '127.0.0.1'
port = 3306
database = 'data_warehouse'

try:
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    with engine.connect() as conn:
        print("[OK] Connected to MySQL successfully!")
except SQLAlchemyError as e:
    print("[ERROR] Failed to connect to MySQL!")
    print(e)
    exit(1)

# ----------------------------
# 2️⃣ Read CSV Files
# ----------------------------
try:
    # Use relative paths from project root
    from pathlib import Path
    base_path = Path(__file__).parent.parent / "data" / "warehouse"
    dim_customer = pd.read_csv(base_path / "dim_customer.csv")
    dim_date = pd.read_csv(base_path / "dim_date.csv")
    dim_product = pd.read_csv(base_path / "dim_product.csv")
    dim_promotion = pd.read_csv(base_path / "dim_promotion.csv")
    dim_store = pd.read_csv(base_path / "dim_store.csv")
    fact_sales = pd.read_csv(base_path / "fact_sales.csv")
    print("[OK] CSV files loaded successfully!")
except Exception as e:
    print("[ERROR] Failed to read CSV files!")
    print(e)
    exit(1)

# ----------------------------
# 3️⃣ Create Tables
# ----------------------------
try:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_customer (
                customer_id INT PRIMARY KEY,
                customer_name VARCHAR(100),
                customer_category VARCHAR(50)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_date (
                date_id INT PRIMARY KEY,
                date DATE,
                year INT,
                month INT,
                day INT,
                weekday VARCHAR(15),
                season VARCHAR(20)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_product (
                product_id INT PRIMARY KEY,
                product VARCHAR(100)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_promotion (
                promotion_id INT PRIMARY KEY,
                promotion VARCHAR(100)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_store (
                store_id INT PRIMARY KEY,
                city VARCHAR(100),
                store_type VARCHAR(50)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fact_sales (
                sales_id INT PRIMARY KEY,
                transaction_id INT,
                date_id INT,
                customer_id INT,
                product_id INT,
                promotion_id INT,
                store_id INT,
                total_items INT,
                total_cost DECIMAL(10,2),
                discount_applied DECIMAL(10,2),
                payment_method VARCHAR(50),
                FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
                FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
                FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
                FOREIGN KEY (promotion_id) REFERENCES dim_promotion(promotion_id),
                FOREIGN KEY (store_id) REFERENCES dim_store(store_id)
            );
        """))
        print("[OK] Tables created or already exist!")
except SQLAlchemyError as e:
    print("[ERROR] Failed to create tables!")
    print(e)
    exit(1)

# ----------------------------
# 4️⃣ Truncate Tables
# ----------------------------
try:
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
        tables = ["fact_sales", "dim_customer", "dim_date", "dim_product", "dim_promotion", "dim_store"]
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE {table};"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
        print("[OK] Tables truncated successfully!")
except SQLAlchemyError as e:
    print("[ERROR] Failed to truncate tables!")
    print(e)
    exit(1)

# ----------------------------
# 5️⃣ Load Data into Tables
# ----------------------------
try:
    dim_customer.to_sql("dim_customer", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] dim_customer loaded")
    dim_date.to_sql("dim_date", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] dim_date loaded")
    dim_product.to_sql("dim_product", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] dim_product loaded")
    dim_promotion.to_sql("dim_promotion", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] dim_promotion loaded")
    dim_store.to_sql("dim_store", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] dim_store loaded")
    fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False, chunksize=1000)
    print("[OK] fact_sales loaded")
    print("[SUCCESS] All CSV files loaded successfully into the star schema database!")
except SQLAlchemyError as e:
    print("[ERROR] Failed to load data into tables!")
    print(e)
    exit(1)
