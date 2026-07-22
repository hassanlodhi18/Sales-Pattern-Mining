"""
Standalone script to display all visualizations
Run this script to see all available visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_style("whitegrid")

def load_data():
    """Load and merge all data files"""
    warehouse_path = Path("data/warehouse")
    fact = pd.read_csv(warehouse_path / "fact_sales.csv")
    dim_customer = pd.read_csv(warehouse_path / "dim_customer.csv")
    dim_date = pd.read_csv(warehouse_path / "dim_date.csv")
    dim_product = pd.read_csv(warehouse_path / "dim_product.csv")
    dim_store = pd.read_csv(warehouse_path / "dim_store.csv")
    dim_promotion = pd.read_csv(warehouse_path / "dim_promotion.csv")

    df = fact.merge(dim_customer, on="customer_id", how="left") \
             .merge(dim_date, on="date_id", how="left")
    # Handle product_key vs product_id column name difference
    if 'product_key' in dim_product.columns:
        df = df.merge(dim_product, left_on="product_id", right_on="product_key", how="left")
    else:
        df = df.merge(dim_product, on="product_id", how="left")
    df = df.merge(dim_store, on="store_id", how="left") \
           .merge(dim_promotion, on="promotion_id", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def main():
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} records")
    
    # Create output directory for saved plots
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # ===================== MATPLOTLIB VISUALIZATIONS =====================
    print("\n1. Creating Daily Sales Trend (Matplotlib)...")
    daily_sales = df.groupby(df["Date"].dt.date)["Total_Cost"].sum()
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily_sales.index, daily_sales.values, linewidth=2, color='#2E86AB')
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Total Sales", fontsize=12)
    ax1.set_title("Daily Sales Trend", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "daily_sales_trend.png", dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved to: {output_dir / 'daily_sales_trend.png'}")
    plt.show()
    
    # ===================== SEABORN HEATMAP =====================
    print("\n2. Creating Monthly Sales Heatmap (Seaborn)...")
    monthly_category = pd.crosstab(
        df["month"],
        df["Customer_Category"],
        values=df["Total_Cost"],
        aggfunc="sum"
    )
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(monthly_category, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2, cbar_kws={'label': 'Total Sales'})
    ax2.set_title("Monthly Sales by Customer Category", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Customer Category", fontsize=12)
    ax2.set_ylabel("Month", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_category_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved to: {output_dir / 'monthly_category_heatmap.png'}")
    plt.show()
    
    # ===================== PLOTLY INTERACTIVE VISUALIZATIONS =====================
    print("\n3. Creating Interactive Monthly Sales Trend (Plotly)...")
    monthly_sales = df.groupby("month")["Total_Cost"].sum()
    fig3 = px.line(
        x=monthly_sales.index,
        y=monthly_sales.values,
        markers=True,
        title="Monthly Sales Trend",
        labels={'x': 'Month', 'y': 'Total Sales'}
    )
    fig3.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Sales",
        template="plotly_white",
        height=500
    )
    fig3.write_html(str(output_dir / "monthly_sales_trend.html"))
    print(f"   [OK] Saved to: {output_dir / 'monthly_sales_trend.html'}")
    fig3.show()
    
    # ===================== PLOTLY DASHBOARD =====================
    print("\n4. Creating Interactive Sales Dashboard (Plotly)...")
    category_sales = df.groupby("Customer_Category")["Total_Cost"].sum()
    store_sales = df.groupby("Store_Type")["Total_Cost"].sum()
    monthly_sales = df.groupby("month")["Total_Cost"].sum()
    city_sales = df.groupby("City")["Total_Cost"].sum().nlargest(10)
    
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Sales by Customer Category",
            "Sales by Store Type",
            "Monthly Sales Trend",
            "Top 10 Cities by Sales"
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    fig4.add_trace(
        go.Bar(x=category_sales.index, y=category_sales.values, marker_color='#2E86AB', name="Category"),
        row=1, col=1
    )
    fig4.add_trace(
        go.Bar(x=store_sales.index, y=store_sales.values, marker_color='#A23B72', name="Store"),
        row=1, col=2
    )
    fig4.add_trace(
        go.Scatter(x=monthly_sales.index, y=monthly_sales.values, mode="lines+markers", 
                  marker_color='#F18F01', name="Monthly"),
        row=2, col=1
    )
    fig4.add_trace(
        go.Bar(x=city_sales.index, y=city_sales.values, marker_color='#C73E1D', name="City"),
        row=2, col=2
    )
    
    fig4.update_layout(
        height=900,
        showlegend=False,
        template="plotly_white",
        title_text="Interactive Sales Dashboard",
        title_x=0.5
    )
    
    fig4.update_xaxes(title_text="Customer Category", row=1, col=1)
    fig4.update_xaxes(title_text="Store Type", row=1, col=2)
    fig4.update_xaxes(title_text="Month", row=2, col=1)
    fig4.update_xaxes(title_text="City", row=2, col=2)
    
    fig4.update_yaxes(title_text="Total Sales", row=1, col=1)
    fig4.update_yaxes(title_text="Total Sales", row=1, col=2)
    fig4.update_yaxes(title_text="Total Sales", row=2, col=1)
    fig4.update_yaxes(title_text="Total Sales", row=2, col=2)
    
    fig4.write_html(str(output_dir / "interactive_dashboard.html"))
    print(f"   [OK] Saved to: {output_dir / 'interactive_dashboard.html'}")
    fig4.show()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"Check the '{output_dir}' directory for saved files.")
    print("="*60)

if __name__ == "__main__":
    main()

