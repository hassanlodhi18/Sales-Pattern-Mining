"""
Exploratory Data Analysis (EDA) on Retail Transactions Warehouse Data
This script performs comprehensive EDA on the data warehouse tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
WAREHOUSE_PATH = Path("data/warehouse")

def load_data():
    """Load all warehouse tables"""
    print("=" * 80)
    print("LOADING DATA FROM WAREHOUSE")
    print("=" * 80)
    
    try:
        fact_sales = pd.read_csv(WAREHOUSE_PATH / "fact_sales.csv")
        dim_customer = pd.read_csv(WAREHOUSE_PATH / "dim_customer.csv")
        dim_date = pd.read_csv(WAREHOUSE_PATH / "dim_date.csv")
        dim_product = pd.read_csv(WAREHOUSE_PATH / "dim_product.csv")
        dim_store = pd.read_csv(WAREHOUSE_PATH / "dim_store.csv")
        dim_promotion = pd.read_csv(WAREHOUSE_PATH / "dim_promotion.csv")
        
        print(f"[OK] Fact Sales: {fact_sales.shape[0]:,} rows, {fact_sales.shape[1]} columns")
        print(f"[OK] Dim Customer: {dim_customer.shape[0]:,} rows, {dim_customer.shape[1]} columns")
        print(f"[OK] Dim Date: {dim_date.shape[0]:,} rows, {dim_date.shape[1]} columns")
        print(f"[OK] Dim Product: {dim_product.shape[0]:,} rows, {dim_product.shape[1]} columns")
        print(f"[OK] Dim Store: {dim_store.shape[0]:,} rows, {dim_store.shape[1]} columns")
        print(f"[OK] Dim Promotion: {dim_promotion.shape[0]:,} rows, {dim_promotion.shape[1]} columns")
        
        return fact_sales, dim_customer, dim_date, dim_product, dim_store, dim_promotion
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def basic_info(df, name):
    """Display basic information about a dataframe"""
    print(f"\n{'=' * 80}")
    print(f"BASIC INFORMATION: {name}")
    print(f"{'=' * 80}")
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")

def analyze_fact_table(fact_sales):
    """Analyze the fact sales table"""
    print(f"\n{'=' * 80}")
    print("FACT SALES TABLE ANALYSIS")
    print(f"{'=' * 80}")
    
    basic_info(fact_sales, "Fact Sales")
    
    # Statistical summary
    print(f"\n{'=' * 80}")
    print("STATISTICAL SUMMARY")
    print(f"{'=' * 80}")
    print(fact_sales.describe())
    
    # Check for numeric columns
    numeric_cols = fact_sales.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    
    return numeric_cols

def analyze_dimensions(dim_customer, dim_date, dim_product, dim_store, dim_promotion):
    """Analyze dimension tables"""
    print(f"\n{'=' * 80}")
    print("DIMENSION TABLES ANALYSIS")
    print(f"{'=' * 80}")
    
    # Customer dimension
    print("\n--- CUSTOMER DIMENSION ---")
    print(f"Unique customers: {dim_customer['customer_id'].nunique():,}")
    print(f"Customer categories distribution:")
    print(dim_customer['Customer_Category'].value_counts())
    
    # Date dimension
    print("\n--- DATE DIMENSION ---")
    print(f"Date range: {dim_date['Date'].min()} to {dim_date['Date'].max()}")
    print(f"Years: {sorted(dim_date['year'].unique())}")
    print(f"Seasons distribution:")
    print(dim_date['season'].value_counts())
    print(f"Month distribution:")
    print(dim_date['month'].value_counts().sort_index())
    
    # Product dimension
    print("\n--- PRODUCT DIMENSION ---")
    print(f"Unique products: {dim_product['Product'].nunique():,}")
    # Check for product_key or product_id column
    product_id_col = 'product_key' if 'product_key' in dim_product.columns else 'product_id'
    print(f"Unique product IDs: {dim_product[product_id_col].nunique():,}")
    print(f"Top 10 products by frequency:")
    print(dim_product['Product'].value_counts().head(10))
    
    # Store dimension
    print("\n--- STORE DIMENSION ---")
    print(f"Unique stores: {dim_store['store_id'].nunique():,}")
    print(f"Cities: {dim_store['City'].nunique()}")
    print(f"Store types: {dim_store['Store_Type'].nunique()}")
    print(f"Store types distribution:")
    print(dim_store['Store_Type'].value_counts())
    print(f"Cities distribution:")
    print(dim_store['City'].value_counts())
    
    # Promotion dimension
    print("\n--- PROMOTION DIMENSION ---")
    print(f"Promotion types:")
    print(dim_promotion['Promotion'].value_counts())

def create_merged_dataset(fact_sales, dim_customer, dim_date, dim_product, dim_store, dim_promotion):
    """Create a merged dataset for comprehensive analysis"""
    print(f"\n{'=' * 80}")
    print("CREATING MERGED DATASET")
    print(f"{'=' * 80}")
    
    # Merge fact with all dimensions
    merged = fact_sales.copy()
    
    merged = merged.merge(dim_customer, on='customer_id', how='left')
    merged = merged.merge(dim_date, on='date_id', how='left')
    # Handle product_key vs product_id column name difference
    if 'product_key' in dim_product.columns:
        merged = merged.merge(dim_product, left_on='product_id', right_on='product_key', how='left')
    else:
        merged = merged.merge(dim_product, on='product_id', how='left')
    merged = merged.merge(dim_store, on='store_id', how='left')
    merged = merged.merge(dim_promotion, on='promotion_id', how='left')
    
    print(f"Merged dataset shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    print(f"\nMerged columns: {list(merged.columns)}")
    
    return merged

def sales_analysis(merged):
    """Perform sales analysis"""
    print(f"\n{'=' * 80}")
    print("SALES ANALYSIS")
    print(f"{'=' * 80}")
    
    # Total sales
    if 'Total_Cost' in merged.columns:
        total_sales = merged['Total_Cost'].sum()
        avg_transaction = merged['Total_Cost'].mean()
        median_transaction = merged['Total_Cost'].median()
        
        print(f"\nTotal Sales: ${total_sales:,.2f}")
        print(f"Average Transaction Value: ${avg_transaction:.2f}")
        print(f"Median Transaction Value: ${median_transaction:.2f}")
        print(f"Total Transactions: {len(merged):,}")
    
    # Sales by customer category
    if 'Customer_Category' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Customer Category ---")
        sales_by_category = merged.groupby('Customer_Category')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_category.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_category = sales_by_category.sort_values('Total_Sales', ascending=False)
        print(sales_by_category)
    
    # Sales by store type
    if 'Store_Type' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Store Type ---")
        sales_by_store = merged.groupby('Store_Type')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_store.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_store = sales_by_store.sort_values('Total_Sales', ascending=False)
        print(sales_by_store)
    
    # Sales by city
    if 'City' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by City ---")
        sales_by_city = merged.groupby('City')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_city.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_city = sales_by_city.sort_values('Total_Sales', ascending=False)
        print(sales_by_city)
    
    # Sales by season
    if 'season' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Season ---")
        sales_by_season = merged.groupby('season')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_season.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_season = sales_by_season.sort_values('Total_Sales', ascending=False)
        print(sales_by_season)
    
    # Sales by promotion
    if 'Promotion' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Promotion ---")
        sales_by_promo = merged.groupby('Promotion')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_promo.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_promo = sales_by_promo.sort_values('Total_Sales', ascending=False)
        print(sales_by_promo)
    
    # Sales by month
    if 'month' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Month ---")
        sales_by_month = merged.groupby('month')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_month.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_month = sales_by_month.sort_index()
        print(sales_by_month)
    
    # Sales by year
    if 'year' in merged.columns and 'Total_Cost' in merged.columns:
        print(f"\n--- Sales by Year ---")
        sales_by_year = merged.groupby('year')['Total_Cost'].agg(['sum', 'mean', 'count'])
        sales_by_year.columns = ['Total_Sales', 'Avg_Transaction', 'Transaction_Count']
        sales_by_year = sales_by_year.sort_index()
        print(sales_by_year)

def create_visualizations(merged):
    """Create visualization plots"""
    print(f"\n{'=' * 80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 80}")
    
    # Create output directory for plots
    output_dir = Path("eda_output")
    output_dir.mkdir(exist_ok=True)
    
    if 'Total_Cost' not in merged.columns:
        print("Total_Cost column not found. Skipping visualizations.")
        return
    
    # 1. Sales by Customer Category
    if 'Customer_Category' in merged.columns:
        plt.figure(figsize=(12, 6))
        sales_by_cat = merged.groupby('Customer_Category')['Total_Cost'].sum().sort_values(ascending=False)
        sales_by_cat.plot(kind='bar', color='steelblue')
        plt.title('Total Sales by Customer Category', fontsize=14, fontweight='bold')
        plt.xlabel('Customer Category', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_customer_category.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_customer_category.png")
        plt.close()
    
    # 2. Sales by Store Type
    if 'Store_Type' in merged.columns:
        plt.figure(figsize=(12, 6))
        sales_by_store = merged.groupby('Store_Type')['Total_Cost'].sum().sort_values(ascending=False)
        sales_by_store.plot(kind='bar', color='coral')
        plt.title('Total Sales by Store Type', fontsize=14, fontweight='bold')
        plt.xlabel('Store Type', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_store_type.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_store_type.png")
        plt.close()
    
    # 3. Sales by City
    if 'City' in merged.columns:
        plt.figure(figsize=(14, 6))
        sales_by_city = merged.groupby('City')['Total_Cost'].sum().sort_values(ascending=False)
        sales_by_city.plot(kind='bar', color='mediumseagreen')
        plt.title('Total Sales by City', fontsize=14, fontweight='bold')
        plt.xlabel('City', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_city.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_city.png")
        plt.close()
    
    # 4. Sales by Season
    if 'season' in merged.columns:
        plt.figure(figsize=(10, 6))
        sales_by_season = merged.groupby('season')['Total_Cost'].sum()
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        sales_by_season = sales_by_season.reindex([s for s in season_order if s in sales_by_season.index])
        sales_by_season.plot(kind='bar', color='gold')
        plt.title('Total Sales by Season', fontsize=14, fontweight='bold')
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_season.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_season.png")
        plt.close()
    
    # 5. Sales by Month
    if 'month' in merged.columns:
        plt.figure(figsize=(14, 6))
        sales_by_month = merged.groupby('month')['Total_Cost'].sum().sort_index()
        sales_by_month.plot(kind='line', marker='o', color='purple', linewidth=2, markersize=8)
        plt.title('Total Sales by Month', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_month.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_month.png")
        plt.close()
    
    # 6. Sales by Year
    if 'year' in merged.columns:
        plt.figure(figsize=(10, 6))
        sales_by_year = merged.groupby('year')['Total_Cost'].sum().sort_index()
        sales_by_year.plot(kind='bar', color='teal')
        plt.title('Total Sales by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'sales_by_year.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: sales_by_year.png")
        plt.close()
    
    # 7. Distribution of Transaction Values
    plt.figure(figsize=(12, 6))
    plt.hist(merged['Total_Cost'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Transaction Values', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(merged['Total_Cost'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ${merged["Total_Cost"].mean():.2f}')
    plt.axvline(merged['Total_Cost'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: ${merged["Total_Cost"].median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'transaction_value_distribution.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: transaction_value_distribution.png")
    plt.close()
    
    # 8. Top Products by Sales
    if 'Product' in merged.columns:
        plt.figure(figsize=(14, 8))
        top_products = merged.groupby('Product')['Total_Cost'].sum().sort_values(ascending=False).head(20)
        top_products.plot(kind='barh', color='indianred')
        plt.title('Top 20 Products by Total Sales', fontsize=14, fontweight='bold')
        plt.xlabel('Total Sales ($)', fontsize=12)
        plt.ylabel('Product', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'top_products_by_sales.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: top_products_by_sales.png")
        plt.close()
    
    print(f"\nAll visualizations saved to '{output_dir}' directory")

def correlation_analysis(merged):
    """Perform correlation analysis on numeric columns"""
    print(f"\n{'=' * 80}")
    print("CORRELATION ANALYSIS")
    print(f"{'=' * 80}")
    
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr_matrix = merged[numeric_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        # Create correlation heatmap
        if len(numeric_cols) > 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_dir = Path("eda_output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: correlation_heatmap.png")
            plt.close()

def main():
    """Main function to run EDA"""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS - RETAIL TRANSACTIONS WAREHOUSE")
    print("=" * 80)
    
    # Load data
    fact_sales, dim_customer, dim_date, dim_product, dim_store, dim_promotion = load_data()
    
    # Analyze fact table
    numeric_cols = analyze_fact_table(fact_sales)
    
    # Analyze dimensions
    analyze_dimensions(dim_customer, dim_date, dim_product, dim_store, dim_promotion)
    
    # Create merged dataset
    merged = create_merged_dataset(fact_sales, dim_customer, dim_date, dim_product, dim_store, dim_promotion)
    
    # Sales analysis
    sales_analysis(merged)
    
    # Create visualizations
    create_visualizations(merged)
    
    # Correlation analysis
    correlation_analysis(merged)
    
    print(f"\n{'=' * 80}")
    print("EDA COMPLETE!")
    print(f"{'=' * 80}")
    print("\nSummary:")
    print(f"- Analyzed {len(merged):,} transactions")
    print(f"- Generated visualizations in 'eda_output' directory")
    print(f"- Check the output above for detailed insights")

if __name__ == "__main__":
    main()

