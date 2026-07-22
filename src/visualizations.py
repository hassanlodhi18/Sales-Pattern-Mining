# visualizations.py
"""
Advanced Visualizations – Streamlit Version
Callable methods for Streamlit integration
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_style("whitegrid")


class AdvancedVisualizations:
    def __init__(self):
        self.df = self.load_data()

    # ===================== DATA LOADING =====================
    @staticmethod
    @st.cache_data
    def load_data():
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

    # ===================== MATPLOTLIB =====================
    def plot_sales_trends_matplotlib(self):
        st.subheader("📈 Daily Sales Trend (Matplotlib)")
        daily_sales = self.df.groupby(self.df["Date"].dt.date)["Total_Cost"].sum()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily_sales.index, daily_sales.values, linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        ax.set_title("Daily Sales Trend")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ===================== SEABORN =====================
    def plot_seaborn_heatmaps(self):
        st.subheader("🔥 Monthly Sales Heatmap (Seaborn)")
        monthly_category = pd.crosstab(
            self.df["month"],
            self.df["Customer_Category"],
            values=self.df["Total_Cost"],
            aggfunc="sum"
        )
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(monthly_category, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2)
        ax2.set_title("Monthly Sales by Customer Category")
        ax2.set_xlabel("Customer Category")
        ax2.set_ylabel("Month")
        st.pyplot(fig2)

    # ===================== PLOTLY SIMPLE TREND =====================
    def plot_interactive_sales_trend(self):
        st.subheader("📊 Monthly Sales Trend (Plotly)")
        monthly_sales = self.df.groupby("month")["Total_Cost"].sum()
        fig = px.line(
            x=monthly_sales.index,
            y=monthly_sales.values,
            markers=True,
            title="Monthly Sales Trend"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales")
        st.plotly_chart(fig, use_container_width=True)

    # ===================== PLOTLY DASHBOARD =====================
    def plot_interactive_dashboard(self):
        st.subheader("📊 Interactive Sales Dashboard (Plotly)")
        category_sales = self.df.groupby("Customer_Category")["Total_Cost"].sum()
        store_sales = self.df.groupby("Store_Type")["Total_Cost"].sum()
        monthly_sales = self.df.groupby("month")["Total_Cost"].sum()
        city_sales = self.df.groupby("City")["Total_Cost"].sum().nlargest(10)

        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Sales by Customer Category",
                "Sales by Store Type",
                "Monthly Sales Trend",
                "Top 10 Cities by Sales"
            ]
        )
        fig3.add_trace(go.Bar(x=category_sales.index, y=category_sales.values), row=1, col=1)
        fig3.add_trace(go.Bar(x=store_sales.index, y=store_sales.values), row=1, col=2)
        fig3.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales.values,
                                  mode="lines+markers"), row=2, col=1)
        fig3.add_trace(go.Bar(x=city_sales.index, y=city_sales.values), row=2, col=2)

        fig3.update_layout(height=900, showlegend=False, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # ===================== 3D PLOT =====================
    # def plot_3d_scatter(self):
    #     st.subheader("🧊 3D Sales Analysis (Plotly)")
    #     agg_3d = self.df.groupby(
    #         ["Customer_Category", "Store_Type", "City"]
    #     ).agg({
    #         "Total_Cost": "sum",
    #         "Total_Items": "sum",
    #         "Transaction_ID": "count"
    #     }).reset_index()

    #     fig4 = px.scatter_3d(
    #         agg_3d,
    #         x="Total_Cost",
    #         y="Total_Items",
    #         z="Transaction_ID",
    #         color="Customer_Category",
    #         size="Total_Cost",
    #         title="3D Sales Analysis"
    #     )
    #     st.plotly_chart(fig4, use_container_width=True)
