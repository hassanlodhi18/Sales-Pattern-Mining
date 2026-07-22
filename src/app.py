"""
Interactive Streamlit GUI Application for Sales Data Warehouse Analytics
Main entry point for the application
"""

import streamlit as st
import pandas as pd
import statsmodels
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from kdd_process import KDDProcess
    from time_series_forecasting import TimeSeriesForecasting
    from association_rules_mining import AssociationRulesMining
    from visualizations import AdvancedVisualizations
    from eda_warehouse import load_data, create_merged_dataset, sales_analysis
    
    # Check for Plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sales Data Warehouse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📈 EDA & Analysis", "🔄 KDD Process", "📉 Time Series Forecasting",
     "🛒 Association Rules", "📊 Visualizations", "🔌 Power BI Report"]
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<div class="main-header">Sales Data Warehouse Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Modules", "6")
    with col2:
        st.metric("Data Warehouse", "✓ Ready")
    with col3:
        st.metric("Status", "🟢 Active")
    
    st.markdown("""
    ## Welcome to Sales Data Warehouse Analytics Platform
    
    This comprehensive platform provides:
    
    ### 🎯 Key Features
    
    1. **Exploratory Data Analysis (EDA)**
       - Comprehensive data analysis
       - Statistical summaries
       - Data quality checks
    
    2. **KDD Process Implementation**
       - Selection, Preprocessing, Transformation
       - Data Mining, Interpretation
    
    3. **Time Series Forecasting**
       - ARIMA, SARIMA, ARMA models
       - Sales trend predictions
    
    4. **Association Rules Mining**
       - Apriori algorithm
       - FP-Growth algorithm
       - Market basket analysis
       - Frequent itemsets and association rules
    
    5. **Advanced Visualizations**
       - Matplotlib, Seaborn, Plotly
       - Interactive dashboards
    
    ### 📁 Data Structure
    
    The application uses a star schema data warehouse:
    - **Fact Table**: fact_sales.csv
    - **Dimension Tables**: 
      - dim_customer.csv
      - dim_date.csv
      - dim_product.csv
      - dim_store.csv
      - dim_promotion.csv
    
    ### 🚀 Getting Started
    
    Navigate through the sidebar to explore different features!
    """)

# ==================== EDA & ANALYSIS PAGE ====================
elif page == "📈 EDA & Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")
    
    if st.button("Load Data", type="primary"):
        with st.spinner("Loading warehouse data..."):
            try:
                fact_sales, dim_customer, dim_date, dim_product, dim_store, dim_promotion = load_data()
                merged = create_merged_dataset(fact_sales, dim_customer, dim_date, 
                                             dim_product, dim_store, dim_promotion)
                st.session_state.merged_data = merged
                st.session_state.data_loaded = True
                st.success("✓ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    if st.session_state.data_loaded:
        merged = st.session_state.merged_data
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = merged['Total_Cost'].sum()
            st.metric("Total Sales", f"${total_sales:,.2f}")
        
        with col2:
            avg_transaction = merged['Total_Cost'].mean()
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
        
        with col3:
            total_transactions = len(merged)
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        with col4:
            unique_customers = merged['Customer_Name'].nunique()
            st.metric("Unique Customers", f"{unique_customers:,}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(merged.head(100), width='stretch')
        
        # Sales analysis
        st.subheader("💰 Sales Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis",
            ["By Customer Category", "By Store Type", "By City", "By Season", "By Month"]
        )
        
        if analysis_type == "By Customer Category":
            sales_by_cat = merged.groupby('Customer_Category')['Total_Cost'].agg(['sum', 'mean', 'count'])
            sales_by_cat.columns = ['Total Sales', 'Avg Transaction', 'Transaction Count']
            st.dataframe(sales_by_cat.sort_values('Total Sales', ascending=False))
        
        elif analysis_type == "By Store Type":
            sales_by_store = merged.groupby('Store_Type')['Total_Cost'].agg(['sum', 'mean', 'count'])
            sales_by_store.columns = ['Total Sales', 'Avg Transaction', 'Transaction Count']
            st.dataframe(sales_by_store.sort_values('Total Sales', ascending=False))
        
        elif analysis_type == "By City":
            sales_by_city = merged.groupby('City')['Total_Cost'].agg(['sum', 'mean', 'count'])
            sales_by_city.columns = ['Total Sales', 'Avg Transaction', 'Transaction Count']
            st.dataframe(sales_by_city.sort_values('Total Sales', ascending=False))
        
        elif analysis_type == "By Season":
            sales_by_season = merged.groupby('season')['Total_Cost'].agg(['sum', 'mean', 'count'])
            sales_by_season.columns = ['Total Sales', 'Avg Transaction', 'Transaction Count']
            st.dataframe(sales_by_season.sort_values('Total Sales', ascending=False))
        
        elif analysis_type == "By Month":
            sales_by_month = merged.groupby('month')['Total_Cost'].agg(['sum', 'mean', 'count'])
            sales_by_month.columns = ['Total Sales', 'Avg Transaction', 'Transaction Count']
            st.dataframe(sales_by_month.sort_index())

# ==================== KDD PROCESS PAGE ====================
elif page == "🔄 KDD Process":
    st.title("🔄 KDD Process Implementation")
    st.markdown("---")
    
    st.markdown("""
    The KDD (Knowledge Discovery in Databases) process consists of 5 steps:
    1. **Selection** - Select relevant data
    2. **Preprocessing** - Clean and prepare data
    3. **Transformation** - Transform data for mining
    4. **Data Mining** - Apply mining techniques
    5. **Interpretation** - Interpret and present results
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.number_input("Sample Size (0 for all)", min_value=0, value=10000, step=1000)
    
    with col2:
        mining_tasks = st.multiselect(
            "Mining Tasks",
            ["summary", "patterns", "correlation", "clustering"],
            default=["summary", "patterns", "correlation", "clustering"]
        )
    
    if st.button("Run KDD Process", type="primary"):
        with st.spinner("Running KDD process..."):
            try:
                kdd = KDDProcess()
                results = kdd.run_complete_process(
                    sample_size=sample_size if sample_size > 0 else None,
                    mining_tasks=mining_tasks
                )
                
                st.success("✓ KDD Process completed!")
                
                # Store results in session state for display
                st.session_state.kdd_results = results
                
                # Display insights
                if 'insights' in results:
                    st.subheader("💡 Insights")
                    for i, insight in enumerate(results['insights'], 1):
                        st.write(f"{i}. {insight}")
                
                # Display summary
                if 'summary' in results:
                    st.subheader("📋 Summary Report")
                    st.json(results['summary'])
                
                # Display mining results
                mining_results = results.get('mining_results', {})
                
                # 0. Summary Statistics
                if 'summary_statistics' in mining_results:
                    st.subheader("📊 0. Summary Statistics")
                    st.dataframe(mining_results['summary_statistics'], width='stretch')
                    with st.expander("📥 Download Summary Statistics"):
                        csv = mining_results['summary_statistics'].to_csv()
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="summary_statistics.csv",
                            mime="text/csv"
                        )
                
                # 1. Category Patterns
                if 'category_patterns' in mining_results:
                    st.subheader("📈 1. Category Patterns")
                    category_df = mining_results['category_patterns']
                    st.dataframe(category_df, width='stretch')
                    with st.expander("📥 Download Category Patterns"):
                        csv = category_df.to_csv()
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="category_patterns.csv",
                            mime="text/csv"
                        )
                
                # 2. Monthly Patterns
                if 'monthly_patterns' in mining_results:
                    st.subheader("📅 2. Monthly Patterns")
                    monthly_df = mining_results['monthly_patterns']
                    st.dataframe(monthly_df, width='stretch')
                    with st.expander("📥 Download Monthly Patterns"):
                        csv = monthly_df.to_csv()
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="monthly_patterns.csv",
                            mime="text/csv"
                        )
                
                # 3. Store Patterns
                if 'store_patterns' in mining_results:
                    st.subheader("🏪 3. Store Patterns")
                    store_df = mining_results['store_patterns']
                    st.dataframe(store_df, width='stretch')
                    with st.expander("📥 Download Store Patterns"):
                        csv = store_df.to_csv()
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="store_patterns.csv",
                            mime="text/csv"
                        )
                
                # 4. Correlation Matrix
                if 'correlation_matrix' in mining_results:
                    st.subheader("🔗 4. Correlation Matrix")
                    corr_df = mining_results['correlation_matrix']
                    st.dataframe(corr_df, width='stretch')
                    
                    # Visualize correlation matrix as heatmap if plotly is available
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = px.imshow(
                                corr_df,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="RdBu",
                                title="Correlation Matrix Heatmap"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass
                    
                    with st.expander("📥 Download Correlation Matrix"):
                        csv = corr_df.to_csv()
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="correlation_matrix.csv",
                            mime="text/csv"
                        )
                
                # 5. Clustering
                if 'clustering' in mining_results:
                    st.subheader("🎯 5. Clustering")
                    clustering_data = mining_results['clustering']
                    
                    # Display cluster information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Clusters", clustering_data.get('n_clusters', 'N/A'))
                    with col2:
                        cluster_sizes = clustering_data.get('cluster_sizes', {})
                        total_points = sum(cluster_sizes.values()) if cluster_sizes else 0
                        st.metric("Total Data Points", total_points)
                    
                    # Display cluster sizes
                    if cluster_sizes:
                        st.write("**Cluster Sizes:**")
                        cluster_sizes_df = pd.DataFrame(
                            list(cluster_sizes.items()),
                            columns=['Cluster', 'Size']
                        )
                        st.dataframe(cluster_sizes_df, width='stretch')
                    
                    # Display clustering details
                    with st.expander("📋 Clustering Details"):
                        st.json({
                            'n_clusters': clustering_data.get('n_clusters'),
                            'cluster_sizes': clustering_data.get('cluster_sizes'),
                            'model_type': 'KMeans'
                        })
                
            except Exception as e:
                st.error(f"Error running KDD process: {e}")
                st.exception(e)

# ==================== TIME SERIES FORECASTING PAGE ====================
elif page == "📉 Time Series Forecasting":
    st.title("📉 Advanced Time Series Forecasting")
    st.markdown("---")
    
    st.markdown("""
    **Interactive forecasting module** with advanced controls for date range selection, frequency, aggregation, 
    and model selection. Forecast sales trends using ARIMA, SARIMA, and ARMA models with confidence intervals.
    """)
    
    # Load data to get date range for date pickers
    try:
        warehouse_path = Path("data/warehouse")
        fact = pd.read_csv(warehouse_path / "fact_sales.csv")
        dim_date = pd.read_csv(warehouse_path / "dim_date.csv")
        merged = fact.merge(dim_date, on='date_id', how='left')
        merged['Date'] = pd.to_datetime(merged['Date'])
        min_date = merged['Date'].min().date()
        max_date = merged['Date'].max().date()
    except Exception as e:
        st.error(f"Error loading data for date range: {e}")
        min_date = pd.Timestamp('2020-01-01').date()
        max_date = pd.Timestamp('2024-12-31').date()
    
    # ========== DATE RANGE SELECTION ==========
    st.subheader("📅 1. Date Range Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            help="Select the start date for historical data filtering"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            help="Select the end date for historical data filtering"
        )
    
    if start_date > end_date:
        st.error("⚠️ Start date must be before end date. Please adjust the dates.")
        st.stop()
    
    # ========== FREQUENCY SELECTION ==========
    st.subheader("⏱️ 2. Frequency Selection")
    frequency = st.selectbox(
        "Time Frequency",
        options=["D", "W", "M"],
        format_func=lambda x: {"D": "Daily (D)", "W": "Weekly (W)", "M": "Monthly (M)"}[x],
        index=0,
        help="Select the time frequency for resampling the data"
    )
    
    # ========== AGGREGATION METHOD ==========
    st.subheader("📊 3. Aggregation Method")
    aggregation = st.selectbox(
        "Aggregation Method",
        options=["sum", "mean", "median"],
        format_func=lambda x: {"sum": "Sum", "mean": "Mean", "median": "Median"}[x],
        index=0,
        help="Select the aggregation method to apply during resampling"
    )
    
    # ========== FORECAST HORIZON ==========
    st.subheader("🔮 4. Forecast Horizon")
    forecast_steps = st.slider(
        "Forecast Steps",
        min_value=7,
        max_value=180,
        value=30,
        step=1,
        help="Number of future periods to forecast (based on selected frequency)"
    )
    st.info(f"📌 Will forecast **{forecast_steps}** {frequency.lower()} periods ahead")
    
    # ========== MODEL SELECTION ==========
    st.subheader("🤖 5. Forecasting Model Selection")
    model_selection = st.multiselect(
        "Select Models to Fit",
        options=["ARIMA", "SARIMA", "ARMA"],
        default=["ARIMA"],
        help="Select one or more forecasting models to fit and compare"
    )
    
    if not model_selection:
        st.warning("⚠️ Please select at least one model to proceed.")
        st.stop()
    
    # ========== SEASONAL CONTROLS (For SARIMA) ==========
    seasonal_period = None
    if "SARIMA" in model_selection:
        st.subheader("🔄 6. Seasonal Controls (For SARIMA)")
        seasonal_period = st.number_input(
            "Seasonal Period",
            min_value=2,
            max_value=52,
            value=12,
            step=1,
            help="Seasonal period (e.g., 7 for weekly, 12 for monthly, 4 for quarterly)"
        )
        st.info(f"📌 SARIMA will use seasonal period of **{seasonal_period}**")
    
    st.markdown("---")
    
    # ========== RUN FORECASTING ==========
    if st.button("🚀 Run Forecasting", type="primary", use_container_width=True):
        with st.spinner("Running time series forecasting..."):
            try:
                # Initialize forecaster with date filtering
                forecaster = TimeSeriesForecasting(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Prepare time series with selected frequency and aggregation
                forecaster.prepare_time_series(frequency=frequency, aggregation=aggregation)
                
                # Display data summary
                st.success(f"✓ Loaded {len(forecaster.data)} records")
                if len(forecaster.data) > 0:
                    st.info(f"📊 Data range: {forecaster.data[forecaster.date_col].min().date()} to {forecaster.data[forecaster.date_col].max().date()}")
                
                # Fit models
                models_fitted = []
                
                if "ARIMA" in model_selection:
                    try:
                        forecaster.fit_arima(order=(1, 1, 1))
                        models_fitted.append("ARIMA")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fit ARIMA model: {e}")
                
                if "SARIMA" in model_selection:
                    try:
                        if seasonal_period is None:
                            seasonal_period = 12  # Default
                        forecaster.fit_sarima(
                            order=(1, 1, 1), 
                            seasonal_order=(1, 1, 1, seasonal_period)
                        )
                        models_fitted.append("SARIMA")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fit SARIMA model: {e}")
                
                if "ARMA" in model_selection:
                    try:
                        # Make data stationary for ARMA if needed
                        if not hasattr(forecaster, 'time_series_stationary'):
                            is_stationary, _ = forecaster.check_stationarity()
                            if not is_stationary:
                                forecaster.make_stationary(method='diff', diff_periods=1)
                        forecaster.fit_arma(order=(1, 1))
                        models_fitted.append("ARMA")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fit ARMA model: {e}")
                
                if not models_fitted:
                    st.error("❌ No models could be fitted. Please check your data and parameters.")
                    st.stop()
                
                st.success(f"✓ Successfully fitted {len(models_fitted)} model(s): {', '.join(models_fitted)}")
                
                # ========== FORECAST OUTPUT METRICS ==========
                st.markdown("---")
                st.subheader("📈 7. Forecast Output Metrics")
                
                # Generate forecasts for each model
                for model_name in models_fitted:
                    if model_name in forecaster.models:
                        try:
                            forecast, ci = forecaster.forecast(
                                model_name=model_name, 
                                steps=forecast_steps, 
                                plot=False
                            )
                            
                            # Display forecast metrics in an expandable section
                            with st.expander(f"📊 {model_name} Forecast Results", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    mean_forecast = forecast.mean()
                                    st.metric(
                                        "Mean Forecast",
                                        f"${mean_forecast:,.2f}",
                                        help="Average forecasted value across all forecast periods"
                                    )
                                
                                with col2:
                                    lower_ci = ci.iloc[:, 0].mean()
                                    st.metric(
                                        "Lower Confidence Interval (95%)",
                                        f"${lower_ci:,.2f}",
                                        help="Average lower bound of 95% confidence interval"
                                    )
                                
                                with col3:
                                    upper_ci = ci.iloc[:, 1].mean()
                                    st.metric(
                                        "Upper Confidence Interval (95%)",
                                        f"${upper_ci:,.2f}",
                                        help="Average upper bound of 95% confidence interval"
                                    )
                                
                                # Detailed forecast table
                                st.markdown("**Detailed Forecast Table:**")
                                # Calculate next period start date based on frequency
                                last_date = forecaster.time_series.index[-1]
                                if frequency == 'D':
                                    next_date = last_date + pd.Timedelta(days=1)
                                elif frequency == 'W':
                                    next_date = last_date + pd.Timedelta(weeks=1)
                                elif frequency == 'M':
                                    next_date = last_date + pd.DateOffset(months=1)
                                else:
                                    next_date = last_date + pd.Timedelta(days=1)
                                
                                forecast_dates = pd.date_range(
                                    start=next_date,
                                    periods=len(forecast),
                                    freq=frequency
                                )
                                
                                forecast_df = pd.DataFrame({
                                    'Date': forecast_dates,
                                    'Mean Forecast ($)': forecast.values,
                                    'Lower CI ($)': ci.iloc[:, 0].values,
                                    'Upper CI ($)': ci.iloc[:, 1].values
                                })
                                st.dataframe(forecast_df, use_container_width=True, height=300)
                                
                                # Download button
                                csv = forecast_df.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download {model_name} Forecast as CSV",
                                    data=csv,
                                    file_name=f"{model_name.lower()}_forecast_{start_date}_{end_date}.csv",
                                    mime="text/csv",
                                    key=f"download_{model_name}"
                                )
                                
                                # Visualize forecast
                                if PLOTLY_AVAILABLE:
                                    try:
                                        fig = go.Figure()
                                        
                                        # Historical data (last 100 periods)
                                        hist_data = forecaster.time_series.tail(100)
                                        fig.add_trace(go.Scatter(
                                            x=hist_data.index,
                                            y=hist_data.values,
                                            mode='lines',
                                            name='Historical Data',
                                            line=dict(color='blue', width=2)
                                        ))
                                        
                                        # Forecast
                                        # Calculate next period start date based on frequency
                                        last_date = forecaster.time_series.index[-1]
                                        if frequency == 'D':
                                            next_date = last_date + pd.Timedelta(days=1)
                                        elif frequency == 'W':
                                            next_date = last_date + pd.Timedelta(weeks=1)
                                        elif frequency == 'M':
                                            next_date = last_date + pd.DateOffset(months=1)
                                        else:
                                            next_date = last_date + pd.Timedelta(days=1)
                                        
                                        forecast_dates = pd.date_range(
                                            start=next_date,
                                            periods=len(forecast),
                                            freq=frequency
                                        )
                                        fig.add_trace(go.Scatter(
                                            x=forecast_dates,
                                            y=forecast.values,
                                            mode='lines',
                                            name=f'{model_name} Forecast',
                                            line=dict(color='red', width=2, dash='dash')
                                        ))
                                        
                                        # Confidence intervals
                                        fig.add_trace(go.Scatter(
                                            x=forecast_dates,
                                            y=ci.iloc[:, 1].values,
                                            mode='lines',
                                            name='Upper CI (95%)',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=forecast_dates,
                                            y=ci.iloc[:, 0].values,
                                            mode='lines',
                                            name='Lower CI (95%)',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            fillcolor='rgba(255,0,0,0.2)',
                                            showlegend=True
                                        ))
                                        
                                        fig.update_layout(
                                            title=f'{model_name} Forecast with Confidence Intervals',
                                            xaxis_title='Date',
                                            yaxis_title='Sales ($)',
                                            hovermode='x unified',
                                            height=500
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not create interactive plot: {e}")
                        
                        except Exception as e:
                            st.error(f"Error generating forecast for {model_name}: {e}")
                
                # ========== MODEL COMPARISON ==========
                if len(forecaster.model_metrics) > 1:
                    st.markdown("---")
                    st.subheader("📊 Model Comparison")
                    comparison_df = pd.DataFrame(forecaster.model_metrics).T
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualize model comparison
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = go.Figure()
                            models_list = list(comparison_df.index)
                            aic_values = comparison_df['AIC'].values
                            
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=aic_values,
                                text=[f'{val:.2f}' for val in aic_values],
                                textposition='auto',
                                marker_color='lightblue'
                            ))
                            
                            fig.update_layout(
                                title='Model Comparison (AIC - Lower is Better)',
                                xaxis_title='Model',
                                yaxis_title='AIC Value',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass
                
                st.success("✅ Forecasting completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error in forecasting: {e}")
                st.exception(e)

# ==================== ASSOCIATION RULES PAGE ====================
elif page == "🛒 Association Rules":
    st.title("🛒 Association Rules Mining")
    st.markdown("---")
    
    st.markdown("""
    **Market Basket Analysis** using Association Rules Mining. Discover frequent itemsets and association rules 
    using **Apriori** or **FP-Growth** algorithms. Find patterns in customer purchasing behavior to drive 
    cross-selling and product recommendations.
    """)
    
    # Initialize session state for association rules
    if 'arm' not in st.session_state:
        st.session_state.arm = None
    if 'association_results' not in st.session_state:
        st.session_state.association_results = None
    
    # ========== ALGORITHM SELECTION ==========
    st.subheader("🤖 1. Algorithm Selection")
    algorithm = st.radio(
        "Select Algorithm",
        options=["Apriori", "FP-Growth"],
        horizontal=True,
        help="Apriori: Traditional frequent itemset mining. FP-Growth: Faster algorithm using FP-tree structure."
    )
    
    st.info(f"📌 Selected Algorithm: **{algorithm}**")
    
    # ========== PARAMETER CONFIGURATION ==========
    st.subheader("⚙️ 2. Parameter Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.001,
            max_value=0.5,
            value=0.02,
            step=0.001,
            format="%.3f",
            help="Minimum support threshold (0.001 = 0.1%, 0.02 = 2%)"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="Minimum confidence threshold for association rules"
        )
    
    with col3:
        sample_size = st.number_input(
            "Sample Size (0 for all)",
            min_value=0,
            value=50000,
            step=5000,
            help="Number of transactions to sample (0 = use all data)"
        )
    
    st.markdown("---")
    
    # ========== RUN ANALYSIS ==========
    if st.button("🚀 Run Association Rules Mining", type="primary", use_container_width=True):
        with st.spinner(f"Running {algorithm} algorithm..."):
            try:
                # Initialize Association Rules Mining
                arm = AssociationRulesMining()
                st.session_state.arm = arm
                
                # Load transactions
                with st.spinner("Loading transactions..."):
                    arm.load_transactions(sample_size=sample_size if sample_size > 0 else None)
                    st.success(f"✓ Loaded {len(arm.transactions):,} transactions")
                
                # Prepare transaction matrix
                with st.spinner("Preparing transaction matrix..."):
                    arm.prepare_transaction_matrix(min_support=min_support)
                    st.success(f"✓ Transaction matrix created: {arm.transaction_matrix.shape}")
                
                # Find frequent itemsets
                algorithm_lower = algorithm.lower().replace('-', '')
                with st.spinner(f"Finding frequent itemsets using {algorithm}..."):
                    arm.find_frequent_itemsets(
                        min_support=min_support,
                        algorithm=algorithm_lower,
                        use_colnames=True
                    )
                    
                    # Check if any frequent itemsets were found
                    if arm.frequent_itemsets is None or len(arm.frequent_itemsets) == 0:
                        st.warning(f"⚠️ No frequent itemsets found with min_support={min_support}")
                        st.info("💡 **Tip**: Try lowering the minimum support threshold (e.g., 0.01 or 0.005)")
                        st.stop()
                    
                    st.success(f"✓ Found {len(arm.frequent_itemsets)} frequent itemsets")
                
                # Generate association rules (only if frequent itemsets exist)
                if arm.frequent_itemsets is not None and len(arm.frequent_itemsets) > 0:
                    with st.spinner("Generating association rules..."):
                        try:
                            arm.generate_association_rules(
                                metric='confidence',
                                min_threshold=min_confidence
                            )
                            if arm.association_rules is not None and len(arm.association_rules) > 0:
                                st.success(f"✓ Generated {len(arm.association_rules)} association rules")
                            else:
                                st.warning(f"⚠️ No association rules found with min_confidence={min_confidence}")
                                st.info("💡 **Tip**: Try lowering the minimum confidence threshold (e.g., 0.3 or 0.4)")
                        except Exception as e:
                            st.error(f"Error generating association rules: {e}")
                            st.info("💡 **Tip**: Try adjusting the parameters (lower min_support or min_confidence)")
                else:
                    st.error("Cannot generate association rules: No frequent itemsets found")
                    st.stop()
                
                # Analyze rules (only if association rules exist)
                insights = {}
                if arm.association_rules is not None and len(arm.association_rules) > 0:
                    with st.spinner("Analyzing rules..."):
                        insights = arm.analyze_rules()
                else:
                    insights = {}
                
                # Store results in session state
                st.session_state.association_results = {
                    'frequent_itemsets': arm.frequent_itemsets,
                    'association_rules': arm.association_rules if arm.association_rules is not None else pd.DataFrame(),
                    'insights': insights,
                    'algorithm': algorithm,
                    'parameters': {
                        'min_support': min_support,
                        'min_confidence': min_confidence,
                        'sample_size': sample_size
                    }
                }
                
                # Show completion message
                if arm.association_rules is not None and len(arm.association_rules) > 0:
                    st.success("✅ Association Rules Mining completed successfully!")
                elif arm.frequent_itemsets is not None and len(arm.frequent_itemsets) > 0:
                    st.success("✅ Frequent itemsets found! (No association rules with current parameters)")
                else:
                    st.warning("⚠️ Analysis completed but no frequent itemsets found with current parameters")
                
            except Exception as e:
                st.error(f"❌ Error in association rules mining: {e}")
                st.exception(e)
    
    # ========== DISPLAY RESULTS ==========
    if st.session_state.association_results is not None:
        results = st.session_state.association_results
        frequent_itemsets = results['frequent_itemsets']
        association_rules = results['association_rules']
        insights = results['insights']
        
        st.markdown("---")
        st.subheader("📊 Results Summary")
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Algorithm", results['algorithm'])
        
        with col2:
            st.metric("Frequent Itemsets", len(frequent_itemsets))
        
        with col3:
            st.metric("Association Rules", len(association_rules) if len(association_rules) > 0 else 0)
        
        with col4:
            if 'high_confidence_rules' in insights:
                st.metric("High Confidence Rules", insights['high_confidence_rules'])
            else:
                st.metric("High Confidence Rules", 0)
        
        # ========== FREQUENT ITEMSETS ==========
        st.subheader("📦 Frequent Itemsets")
        
        # Format frequent itemsets for display
        itemsets_display = frequent_itemsets.copy()
        itemsets_display['Items'] = itemsets_display['itemsets'].apply(
            lambda x: ', '.join(sorted(list(x)))
        )
        itemsets_display['Itemset Size'] = itemsets_display['itemsets'].apply(len)
        
        # Display columns
        display_cols = ['Items', 'Itemset Size', 'support']
        itemsets_display = itemsets_display[display_cols].sort_values('support', ascending=False)
        
        st.dataframe(
            itemsets_display,
            width='stretch',
            height=400
        )
        
        # Itemset size distribution
        if len(itemsets_display) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Itemset Size Distribution**")
                size_counts = itemsets_display['Itemset Size'].value_counts().sort_index()
                try:
                    st.bar_chart(size_counts)
                except (SyntaxError, ImportError, OSError) as e:
                    # Fallback to DataFrame display if chart fails (e.g., altair import issues)
                    size_counts_df = pd.DataFrame({
                        'Itemset Size': size_counts.index,
                        'Count': size_counts.values
                    })
                    st.dataframe(size_counts_df, width='stretch', hide_index=True)
            
            with col2:
                st.write("**Top 10 Itemsets by Support**")
                top_itemsets = itemsets_display.head(10)
                for idx, row in top_itemsets.iterrows():
                    st.write(f"• {row['Items']}: {row['support']:.4f}")
        
        # ========== ASSOCIATION RULES ==========
        if len(association_rules) > 0:
            st.subheader("🔗 Association Rules")
            
            # Format association rules for display
            rules_display = association_rules.copy()
            rules_display['Antecedents'] = rules_display['antecedents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_display['Consequents'] = rules_display['consequents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_display['Rule'] = rules_display.apply(
                lambda row: f"{row['Antecedents']} → {row['Consequents']}", axis=1
            )
            
            # Display columns
            rule_display_cols = ['Rule', 'support', 'confidence', 'lift']
            if 'conviction' in rules_display.columns:
                rule_display_cols.append('conviction')
            if 'leverage' in rules_display.columns:
                rule_display_cols.append('leverage')
            
            rules_display = rules_display[rule_display_cols].sort_values('confidence', ascending=False)
            
            st.dataframe(
                rules_display,
                width='stretch',
                height=400
            )
            
            # Rule metrics visualization
            if len(rules_display) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 10 Rules by Confidence**")
                    top_rules = rules_display.head(10)
                    for idx, row in top_rules.iterrows():
                        st.write(f"• {row['Rule']}")
                        st.write(f"  Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
                
                with col2:
                    st.write("**Rules Statistics**")
                    st.metric("Avg Confidence", f"{rules_display['confidence'].mean():.4f}")
                    st.metric("Max Confidence", f"{rules_display['confidence'].max():.4f}")
                    st.metric("Avg Lift", f"{rules_display['lift'].mean():.4f}")
                    st.metric("Max Lift", f"{rules_display['lift'].max():.4f}")
        
        else:
            st.warning("⚠️ No association rules found with the current parameters. Try lowering min_confidence or min_support.")
        
        # ========== INSIGHTS ==========
        if insights and len(insights) > 0:
            st.subheader("💡 Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'high_confidence_rules' in insights:
                    st.metric("High Confidence Rules (≥0.7)", insights['high_confidence_rules'])
                if 'high_lift_rules' in insights:
                    st.metric("High Lift Rules (≥2.0)", insights['high_lift_rules'])
                
                if 'top_antecedents' in insights:
                    st.write("**Top 5 Antecedents**")
                    for item, count in insights['top_antecedents']:
                        st.write(f"• {item}: {count} rules")
            
            with col2:
                if 'top_consequents' in insights:
                    st.write("**Top 5 Consequents**")
                    for item, count in insights['top_consequents']:
                        st.write(f"• {item}: {count} rules")
        
        # ========== DOWNLOAD RESULTS ==========
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download frequent itemsets
            if frequent_itemsets is not None:
                itemsets_csv = frequent_itemsets.copy()
                itemsets_csv['itemsets'] = itemsets_csv['itemsets'].apply(
                    lambda x: ', '.join(sorted(list(x)))
                )
                csv_itemsets = itemsets_csv.to_csv(index=False)
                st.download_button(
                    label="Download Frequent Itemsets (CSV)",
                    data=csv_itemsets,
                    file_name=f"frequent_itemsets_{algorithm.lower()}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Download association rules
            if len(association_rules) > 0:
                rules_csv = association_rules.copy()
                rules_csv['antecedents'] = rules_csv['antecedents'].apply(
                    lambda x: ', '.join(sorted(list(x)))
                )
                rules_csv['consequents'] = rules_csv['consequents'].apply(
                    lambda x: ', '.join(sorted(list(x)))
                )
                csv_rules = rules_csv.to_csv(index=False)
                st.download_button(
                    label="Download Association Rules (CSV)",
                    data=csv_rules,
                    file_name=f"association_rules_{algorithm.lower()}.csv",
                    mime="text/csv"
                )
        
        # Save all results button
        if st.button("💾 Save All Results to Files", use_container_width=True):
            try:
                arm = st.session_state.arm
                if arm:
                    output_dir = f"association_rules_output_{algorithm.lower()}"
                    arm.save_results(output_dir=output_dir)
                    st.success(f"✅ Results saved to {output_dir}/ directory!")
            except Exception as e:
                st.error(f"Error saving results: {e}")

# ==================== VISUALIZATIONS PAGE ====================
elif page == "📊 Visualizations":
    st.title("📊 Advanced Visualizations")
    st.markdown("---")
    
    # Check if KDD results are available
    kdd_results_available = 'kdd_results' in st.session_state and st.session_state.kdd_results is not None
    
    if kdd_results_available:
        st.info("✅ **KDD Results Available:** You can visualize KDD mining results below, or use standard visualizations.")
        st.markdown("---")
    
    # Visualization source selection
    if kdd_results_available:
        viz_source = st.radio(
            "Select Visualization Source",
            ["📊 KDD Results", "📈 Standard Visualizations"],
            help="Choose to visualize KDD mining results or standard data visualizations"
        )
        st.markdown("---")
    else:
        viz_source = "📈 Standard Visualizations"
        st.info("💡 **Tip:** Run the KDD Process first to visualize mining results here.")
        st.markdown("---")
    
    # ========== KDD RESULTS VISUALIZATIONS ==========
    if viz_source == "📊 KDD Results" and kdd_results_available:
        st.subheader("📊 KDD Mining Results Visualizations")
        
        kdd_results = st.session_state.kdd_results
        mining_results = kdd_results.get('mining_results', {})
        
        if not mining_results:
            st.warning("⚠️ No mining results found in KDD results. Please run KDD Process with mining tasks enabled.")
        else:
            # 1. Category Patterns Visualization
            if 'category_patterns' in mining_results:
                st.subheader("📈 Category Patterns")
                category_df = mining_results['category_patterns'].copy()
                
                # Handle multi-level columns (flatten if needed)
                if isinstance(category_df.columns, pd.MultiIndex):
                    category_df.columns = ['_'.join(map(str, col)).strip() for col in category_df.columns.values]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if PLOTLY_AVAILABLE:
                        # Reset index to get category names as a column
                        category_df_viz = category_df.reset_index()
                        x_col = category_df_viz.columns[0]  # First column is the category
                        
                        # Find Total_Cost sum column (most common pattern)
                        total_cost_cols = [col for col in category_df_viz.columns if 'Total_Cost' in str(col) and 'sum' in str(col).lower()]
                        if total_cost_cols:
                            y_col = total_cost_cols[0]
                        else:
                            # Use first numeric column
                            numeric_cols = category_df_viz.select_dtypes(include=[np.number]).columns
                            y_col = numeric_cols[0] if len(numeric_cols) > 0 else category_df_viz.columns[1]
                        
                        fig = px.bar(
                            category_df_viz,
                            x=x_col,
                            y=y_col,
                            title="Sales by Customer Category",
                            labels={x_col: 'Customer Category', y_col: 'Total Sales'},
                            color=y_col,
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(category_df, use_container_width=True)
                
                with col2:
                    st.dataframe(category_df, use_container_width=True, height=300)
            
            # 2. Monthly Patterns Visualization
            if 'monthly_patterns' in mining_results:
                st.subheader("📅 Monthly Patterns")
                monthly_df = mining_results['monthly_patterns'].copy()
                
                # Handle multi-level columns (flatten if needed)
                if isinstance(monthly_df.columns, pd.MultiIndex):
                    monthly_df.columns = ['_'.join(map(str, col)).strip() for col in monthly_df.columns.values]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if PLOTLY_AVAILABLE:
                        # Reset index to get month as a column
                        monthly_df_viz = monthly_df.reset_index()
                        x_col = monthly_df_viz.columns[0]  # First column is the month
                        
                        # Find Total_Cost sum column
                        total_cost_cols = [col for col in monthly_df_viz.columns if 'Total_Cost' in str(col) and 'sum' in str(col).lower()]
                        if total_cost_cols:
                            y_col = total_cost_cols[0]
                        else:
                            # Use first numeric column
                            numeric_cols = monthly_df_viz.select_dtypes(include=[np.number]).columns
                            y_col = numeric_cols[0] if len(numeric_cols) > 0 else monthly_df_viz.columns[1]
                        
                        fig = px.line(
                            monthly_df_viz,
                            x=x_col,
                            y=y_col,
                            title="Monthly Sales Trend",
                            labels={x_col: 'Month', y_col: 'Total Sales'},
                            markers=True
                        )
                        fig.update_traces(line=dict(width=3))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(monthly_df, use_container_width=True)
                
                with col2:
                    st.dataframe(monthly_df, use_container_width=True, height=300)
            
            # 3. Store Patterns Visualization
            if 'store_patterns' in mining_results:
                st.subheader("🏪 Store Patterns")
                store_df = mining_results['store_patterns'].copy()
                
                # Handle multi-level columns (flatten if needed)
                if isinstance(store_df.columns, pd.MultiIndex):
                    store_df.columns = ['_'.join(map(str, col)).strip() for col in store_df.columns.values]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if PLOTLY_AVAILABLE:
                        # Reset index to get store type as a column
                        store_df_viz = store_df.reset_index()
                        x_col = store_df_viz.columns[0]  # First column is the store type
                        
                        # Find Total_Cost sum column
                        total_cost_cols = [col for col in store_df_viz.columns if 'Total_Cost' in str(col) and 'sum' in str(col).lower()]
                        if total_cost_cols:
                            y_col = total_cost_cols[0]
                        else:
                            # Use first numeric column
                            numeric_cols = store_df_viz.select_dtypes(include=[np.number]).columns
                            y_col = numeric_cols[0] if len(numeric_cols) > 0 else store_df_viz.columns[1]
                        
                        fig = px.bar(
                            store_df_viz,
                            x=x_col,
                            y=y_col,
                            title="Sales by Store Type",
                            labels={x_col: 'Store Type', y_col: 'Total Sales'},
                            color=y_col,
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(store_df, use_container_width=True)
                
                with col2:
                    st.dataframe(store_df, use_container_width=True, height=300)
            
            # 4. Correlation Matrix Visualization
            if 'correlation_matrix' in mining_results:
                st.subheader("🔗 Correlation Matrix")
                corr_df = mining_results['correlation_matrix']
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = px.imshow(
                                corr_df,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="RdBu",
                                title="Correlation Matrix Heatmap",
                                labels=dict(color="Correlation")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating correlation heatmap: {e}")
                            st.dataframe(corr_df, use_container_width=True)
                    else:
                        st.dataframe(corr_df, use_container_width=True)
                
                with col2:
                    st.dataframe(corr_df, use_container_width=True, height=300)
            
            # 5. Clustering Visualization
            if 'clustering' in mining_results:
                st.subheader("🎯 Clustering Results")
                clustering_data = mining_results['clustering']
                
                col1, col2 = st.columns(2)
                with col1:
                    if PLOTLY_AVAILABLE:
                        cluster_sizes = clustering_data.get('cluster_sizes', {})
                        if cluster_sizes:
                            cluster_df = pd.DataFrame(
                                list(cluster_sizes.items()),
                                columns=['Cluster', 'Size']
                            )
                            fig = px.bar(
                                cluster_df,
                                x='Cluster',
                                y='Size',
                                title="Cluster Sizes Distribution",
                                labels={'Cluster': 'Cluster ID', 'Size': 'Number of Points'},
                                color='Size',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No cluster sizes data available")
                    else:
                        cluster_sizes = clustering_data.get('cluster_sizes', {})
                        if cluster_sizes:
                            cluster_df = pd.DataFrame(
                                list(cluster_sizes.items()),
                                columns=['Cluster', 'Size']
                            )
                            st.dataframe(cluster_df, use_container_width=True)
                
                with col2:
                    st.metric("Number of Clusters", clustering_data.get('n_clusters', 'N/A'))
                    cluster_sizes = clustering_data.get('cluster_sizes', {})
                    total_points = sum(cluster_sizes.values()) if cluster_sizes else 0
                    st.metric("Total Data Points", total_points)
                    
                    with st.expander("📋 Clustering Details"):
                        st.json({
                            'n_clusters': clustering_data.get('n_clusters'),
                            'cluster_sizes': clustering_data.get('cluster_sizes'),
                            'model_type': 'KMeans'
                        })
            
            # ========== OBJECTIVE-FOCUSED VISUALIZATIONS ==========
            st.markdown("---")
            st.subheader("🎯 Objective-Focused Analysis")
            
            # Load warehouse data (fact table and dimensions separately for memory efficiency)
            warehouse_path = Path("data/warehouse")
            fact_sales = None
            dim_product = None
            dim_date = None
            dim_store = None
            
            try:
                # Load fact table
                fact_sales = pd.read_csv(warehouse_path / "fact_sales.csv")
                st.info(f"✓ Loaded fact_sales: {len(fact_sales):,} rows")
                
                # Load dimensions as needed
                dim_product = pd.read_csv(warehouse_path / "dim_product.csv")
                dim_date = pd.read_csv(warehouse_path / "dim_date.csv")
                dim_store = pd.read_csv(warehouse_path / "dim_store.csv")
                
            except Exception as e:
                st.error(f"Error loading warehouse data: {e}")
                st.info("Please ensure warehouse data is available in the data/warehouse folder.")
            
            if fact_sales is not None and not fact_sales.empty:
                # 1. Product Category Sales Distribution Analysis
                st.markdown("### 📦 Objective 1: Product Category Sales Distribution")
                st.markdown("**Goal:** Analyze sales distribution across product categories to identify high-performing categories based on total cost and item volume.")
                
                # Check for product_id column in fact_sales
                product_id_col = None
                for col in ['product_id', 'Product_ID', 'product_ID', 'Product_id', 'product_key', 'Product_Key']:
                    if col in fact_sales.columns:
                        product_id_col = col
                        break
                
                if dim_product is not None and product_id_col:
                    if PLOTLY_AVAILABLE:
                        try:
                            # Aggregate fact table by product_id first (memory efficient)
                            product_agg = fact_sales.groupby(product_id_col).agg({
                                'Total_Cost': ['sum', 'mean'],
                                'Total_Items': ['sum', 'mean']
                            }).reset_index()
                            
                            # Flatten column names
                            product_agg.columns = [product_id_col, 'Total_Cost_sum', 'Total_Cost_mean', 'Total_Items_sum', 'Total_Items_mean']
                            
                            # Merge with product dimension to get category info
                            # Find product key column in dim_product
                            dim_product_key_col = None
                            for col in ['product_key', 'Product_Key', 'product_id', 'Product_ID', 'product_ID', 'Product_id']:
                                if col in dim_product.columns:
                                    dim_product_key_col = col
                                    break
                            
                            # Find product category column in dim_product
                            product_col = None
                            for col in ['Product_Category', 'Category', 'Product', 'Product_Name']:
                                if col in dim_product.columns:
                                    product_col = col
                                    break
                            
                            if product_col and dim_product_key_col:
                                product_analysis = product_agg.merge(
                                    dim_product[[dim_product_key_col, product_col]], 
                                    left_on=product_id_col,
                                    right_on=dim_product_key_col,
                                    how='left'
                                )
                                
                                # Drop key columns for visualization
                                if product_id_col in product_analysis.columns:
                                    product_analysis = product_analysis.drop(product_id_col, axis=1)
                                if dim_product_key_col in product_analysis.columns and dim_product_key_col != product_id_col:
                                    product_analysis = product_analysis.drop(dim_product_key_col, axis=1)
                            else:
                                # Use product_id if no category column found
                                product_analysis = product_agg
                                product_col = product_id_col
                            
                        except Exception as e:
                            st.error(f"Error creating product analysis: {e}")
                            product_analysis = None
                            product_col = None
                    else:
                        product_analysis = None
                        product_col = None
                else:
                    product_analysis = None
                    product_col = None
                
                if product_analysis is not None and not product_analysis.empty and product_col:
                    # Get the actual product column name
                    actual_product_col = product_col
                    
                    # Sort by total cost
                    product_analysis = product_analysis.sort_values('Total_Cost_sum', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Total Cost by Product Category
                            fig1 = px.bar(
                                product_analysis.head(15),  # Top 15 categories
                                x=actual_product_col,
                                y='Total_Cost_sum',
                                title="Total Sales by Product Category",
                                labels={actual_product_col: 'Product Category', 'Total_Cost_sum': 'Total Sales ($)'},
                                color='Total_Cost_sum',
                                color_continuous_scale='Viridis'
                            )
                            fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
                            fig1.update_xaxes(tickmode='linear')
                            st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Total Items by Product Category
                        fig2 = px.bar(
                            product_analysis.head(15),  # Top 15 categories
                            x=actual_product_col,
                            y='Total_Items_sum',
                            title="Total Items Sold by Product Category",
                            labels={actual_product_col: 'Product Category', 'Total_Items_sum': 'Total Items'},
                            color='Total_Items_sum',
                            color_continuous_scale='Plasma'
                        )
                        fig2.update_layout(showlegend=False, xaxis_tickangle=-45)
                        fig2.update_xaxes(tickmode='linear')
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Combined scatter plot: Total Cost vs Total Items
                    st.markdown("**Product Category Performance Matrix:**")
                    fig3 = px.scatter(
                        product_analysis,
                        x='Total_Items_sum',
                        y='Total_Cost_sum',
                        size='Total_Cost_sum',
                        hover_name=actual_product_col,
                        title="Product Category Performance: Total Cost vs Item Volume",
                        labels={'Total_Items_sum': 'Total Items Sold', 'Total_Cost_sum': 'Total Sales ($)'},
                        color='Total_Cost_sum',
                        color_continuous_scale='Blues'
                    )
                    fig3.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Summary table
                    with st.expander("📊 Product Category Summary Table"):
                        st.dataframe(product_analysis, use_container_width=True)
                else:
                    if not PLOTLY_AVAILABLE:
                        st.info("Plotly is required for product category visualizations. Please install with: pip install plotly")
                    else:
                        st.info("Product information not available in warehouse data.")
                
                # 2. Enhanced Monthly Trends with Seasonality
                st.markdown("---")
                st.markdown("### 📅 Objective 2: Month-wise Sales Trends and Seasonality")
                st.markdown("**Goal:** Examine month-wise sales trends and seasonality to understand fluctuations in total cost and transaction volume.")
                
                if dim_date is not None and 'date_id' in fact_sales.columns:
                    if PLOTLY_AVAILABLE:
                        try:
                            # Aggregate fact table by date_id first (memory efficient)
                            date_agg = fact_sales.groupby('date_id').agg({
                                'Total_Cost': ['sum', 'mean'],
                                'Transaction_ID': 'count' if 'Transaction_ID' in fact_sales.columns else 'size'
                            }).reset_index()
                            
                            # Flatten column names
                            transaction_col_name = 'Transaction_ID_count' if 'Transaction_ID' in fact_sales.columns else 'size'
                            date_agg.columns = ['date_id', 'Total_Cost_sum', 'Total_Cost_mean', transaction_col_name]
                            
                            # Merge with date dimension to get month info
                            # Check for month column in dim_date
                            month_col = None
                            date_col = None
                            for col in ['Month', 'month', 'Month_Name']:
                                if col in dim_date.columns:
                                    month_col = col
                                    break
                            if not month_col and 'Date' in dim_date.columns:
                                date_col = 'Date'
                            
                            if month_col:
                                monthly_analysis = date_agg.merge(
                                    dim_date[['date_id', month_col]], 
                                    on='date_id', 
                                    how='left'
                                ).drop('date_id', axis=1)
                                
                                # Group by month to aggregate if multiple dates per month
                                monthly_analysis = monthly_analysis.groupby(month_col).agg({
                                    'Total_Cost_sum': 'sum',
                                    'Total_Cost_mean': 'mean',
                                    transaction_col_name: 'sum'
                                }).reset_index()
                                
                                actual_month_col = month_col
                            elif date_col:
                                monthly_analysis = date_agg.merge(
                                    dim_date[['date_id', date_col]], 
                                    on='date_id', 
                                    how='left'
                                ).drop('date_id', axis=1)
                                
                                # Extract month from date
                                monthly_analysis[date_col] = pd.to_datetime(monthly_analysis[date_col])
                                monthly_analysis['month'] = monthly_analysis[date_col].dt.to_period('M').astype(str)
                                
                                monthly_analysis = monthly_analysis.groupby('month').agg({
                                    'Total_Cost_sum': 'sum',
                                    'Total_Cost_mean': 'mean',
                                    transaction_col_name: 'sum'
                                }).reset_index()
                                
                                actual_month_col = 'month'
                            else:
                                monthly_analysis = None
                                actual_month_col = None
                                
                        except Exception as e:
                            st.error(f"Error creating monthly analysis: {e}")
                            monthly_analysis = None
                            actual_month_col = None
                        
                        if monthly_analysis is not None and actual_month_col:
                            monthly_analysis = monthly_analysis.sort_values(actual_month_col)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Total Cost Trend
                                fig1 = px.line(
                                    monthly_analysis,
                                    x=actual_month_col,
                                    y='Total_Cost_sum',
                                    title="Monthly Total Sales Trend",
                                    labels={actual_month_col: 'Month', 'Total_Cost_sum': 'Total Sales ($)'},
                                    markers=True
                                )
                                fig1.update_traces(line=dict(width=3, color='#1f77b4'))
                                fig1.add_scatter(
                                    x=monthly_analysis[actual_month_col],
                                    y=monthly_analysis['Total_Cost_sum'],
                                    mode='markers',
                                    marker=dict(size=10, color='#ff7f0e'),
                                    name='Monthly Sales'
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Transaction Volume Trend
                                fig2 = px.bar(
                                    monthly_analysis,
                                    x=actual_month_col,
                                    y=transaction_col_name,
                                    title="Monthly Transaction Volume",
                                    labels={actual_month_col: 'Month', transaction_col_name: 'Number of Transactions'},
                                    color=transaction_col_name,
                                    color_continuous_scale='Greens'
                                )
                                fig2.update_layout(showlegend=False)
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Combined view: Dual Y-axis
                            st.markdown("**Combined Monthly Analysis:**")
                            from plotly.subplots import make_subplots
                            
                            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            fig3.add_trace(
                                go.Scatter(
                                    x=monthly_analysis[actual_month_col],
                                    y=monthly_analysis['Total_Cost_sum'],
                                    name="Total Sales ($)",
                                    line=dict(color='#1f77b4', width=3)
                                ),
                                secondary_y=False,
                            )
                            
                            fig3.add_trace(
                                go.Bar(
                                    x=monthly_analysis[actual_month_col],
                                    y=monthly_analysis[transaction_col_name],
                                    name="Transaction Count",
                                    marker_color='#2ca02c',
                                    opacity=0.6
                                ),
                                secondary_y=True,
                            )
                            
                            fig3.update_xaxes(title_text="Month")
                            fig3.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
                            fig3.update_yaxes(title_text="Transaction Count", secondary_y=True)
                            fig3.update_layout(
                                title="Monthly Sales & Transaction Volume Trends",
                                height=500
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                            
                            # Seasonality analysis if season column exists in dim_date
                            if dim_date is not None:
                                season_col = None
                                for col in ['Season', 'season', 'Quarter', 'quarter']:
                                    if col in dim_date.columns:
                                        season_col = col
                                        break
                                
                                if season_col:
                                    try:
                                        # Aggregate by season
                                        season_agg = fact_sales.merge(
                                            dim_date[['date_id', season_col]], 
                                            on='date_id', 
                                            how='left'
                                        ).groupby(season_col).agg({
                                            'Total_Cost': 'sum'
                                        }).reset_index()
                                        
                                        st.markdown("**Seasonal Sales Distribution:**")
                                        fig4 = px.bar(
                                            season_agg,
                                            x=season_col,
                                            y='Total_Cost',
                                            title="Sales by Season",
                                            labels={season_col: 'Season', 'Total_Cost': 'Total Sales ($)'},
                                            color='Total_Cost',
                                            color_continuous_scale='Reds'
                                        )
                                        fig4.update_layout(showlegend=False)
                                        st.plotly_chart(fig4, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not create seasonality analysis: {e}")
                            
                            # Summary table
                            with st.expander("📊 Monthly Trends Summary Table"):
                                st.dataframe(monthly_analysis, use_container_width=True)
                        else:
                            st.info("Month information not available in warehouse data.")
                    else:
                        st.info("Plotly is required for monthly trend visualizations.")
                else:
                    st.info("Date information not available in warehouse data.")
                
                # 3. Store-Level Performance Analysis
                st.markdown("---")
                st.markdown("### 🏪 Objective 3: Store-Level Sales Performance")
                st.markdown("**Goal:** Evaluate and compare store-level sales performance by analyzing total cost contributions.")
                
                if dim_store is not None and 'store_id' in fact_sales.columns:
                    if PLOTLY_AVAILABLE:
                        try:
                            # Aggregate fact table by store_id first (memory efficient)
                            store_agg = fact_sales.groupby('store_id').agg({
                                'Total_Cost': ['sum', 'mean', 'count'],
                                'Total_Items': 'sum'
                            }).reset_index()
                            
                            # Flatten column names
                            store_agg.columns = ['store_id', 'Total_Cost_sum', 'Total_Cost_mean', 'Total_Cost_count', 'Total_Items_sum']
                            
                            # Merge with store dimension to get store info
                            # Find store identifier column in dim_store
                            store_col = None
                            for col in ['City', 'Store_Name', 'Store', 'Store_Type']:
                                if col in dim_store.columns:
                                    store_col = col
                                    break
                            
                            if store_col:
                                store_analysis = store_agg.merge(
                                    dim_store[['store_id', store_col]], 
                                    on='store_id', 
                                    how='left'
                                ).drop('store_id', axis=1)
                                
                                actual_store_col = store_col
                            else:
                                # Use store_id if no name column found
                                store_analysis = store_agg
                                actual_store_col = 'store_id'
                            
                            # Sort by total cost
                            store_analysis = store_analysis.sort_values('Total_Cost_sum', ascending=False)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Top performing stores
                                st.markdown("**Top Performing Stores:**")
                                top_stores = store_analysis.head(20)  # Top 20 stores
                                fig1 = px.bar(
                                    top_stores,
                                    x=actual_store_col,
                                    y='Total_Cost_sum',
                                    title="Top 20 Stores by Total Sales",
                                    labels={actual_store_col: 'Store', 'Total_Cost_sum': 'Total Sales ($)'},
                                    color='Total_Cost_sum',
                                    color_continuous_scale='Greens'
                                )
                                fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
                                fig1.update_xaxes(tickmode='linear')
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Store performance distribution
                                fig2 = px.histogram(
                                    store_analysis,
                                    x='Total_Cost_sum',
                                    nbins=30,
                                    title="Store Sales Distribution",
                                    labels={'Total_Cost_sum': 'Total Sales ($)', 'count': 'Number of Stores'},
                                    color_discrete_sequence=['#2ca02c']
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Performance comparison: Top vs Bottom stores
                            st.markdown("**Store Performance Comparison:**")
                            if len(store_analysis) >= 10:
                                top_10 = store_analysis.head(10)
                                bottom_10 = store_analysis.tail(10)
                                
                                comparison_df = pd.concat([
                                    top_10.assign(Performance='Top 10'),
                                    bottom_10.assign(Performance='Bottom 10')
                                ])
                                
                                fig3 = px.bar(
                                    comparison_df,
                                    x=actual_store_col,
                                    y='Total_Cost_sum',
                                    color='Performance',
                                    title="Top 10 vs Bottom 10 Stores Performance",
                                    labels={actual_store_col: 'Store', 'Total_Cost_sum': 'Total Sales ($)'},
                                    color_discrete_map={'Top 10': '#2ca02c', 'Bottom 10': '#d62728'}
                                )
                                fig3.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Stores", len(store_analysis))
                            with col2:
                                st.metric("Top Store Sales", f"${store_analysis['Total_Cost_sum'].max():,.2f}")
                            with col3:
                                st.metric("Average Store Sales", f"${store_analysis['Total_Cost_sum'].mean():,.2f}")
                            with col4:
                                st.metric("Total Sales (All Stores)", f"${store_analysis['Total_Cost_sum'].sum():,.2f}")
                            
                            # Summary table
                            with st.expander("📊 Store Performance Summary Table"):
                                st.dataframe(store_analysis, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating store analysis: {e}")
                    else:
                        st.info("Plotly is required for store performance visualizations.")
                else:
                    st.info("Store information not available in warehouse data.")
            else:
                st.warning("⚠️ Warehouse data not available. Please ensure warehouse data is in the data/warehouse folder.")
            
            # Summary Statistics Table
            if 'summary_statistics' in mining_results:
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                summary_df = mining_results['summary_statistics']
                st.dataframe(summary_df, use_container_width=True)
    
    # ========== STANDARD VISUALIZATIONS ==========
    elif viz_source == "📈 Standard Visualizations":
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Matplotlib/Seaborn", "Plotly Interactive", "All Visualizations"]
        )
        
        if st.button("Generate Visualizations", type="primary"):
            with st.spinner("Creating visualizations..."):
                try:
                    viz = AdvancedVisualizations()
                    
                    if viz_type in ["Matplotlib/Seaborn", "All Visualizations"]:
                        viz.plot_sales_trends_matplotlib()
                        viz.plot_seaborn_heatmaps()
                        st.success("✓ Matplotlib/Seaborn visualizations created!")
                    
                    if viz_type in ["Plotly Interactive", "All Visualizations"]:
                        if PLOTLY_AVAILABLE:
                            viz.plot_interactive_sales_trend()
                            viz.plot_interactive_dashboard()
                        #    viz.plot_3d_scatter()
                            st.success("✓ Plotly interactive visualizations created!")
                        else:
                            st.warning("Plotly not available. Install with: pip install plotly")
                    
                    st.info("Visualizations saved to 'visualizations' directory")
                    
                except Exception as e:
                    st.error(f"Error creating visualizations: {e}")

# ==================== POWER BI REPORT PAGE ====================
elif page == "🔌 Power BI Report":
    st.title("🔌 Power BI Report")
    st.markdown("---")
    
    # Power BI embedded report URL
    powerbi_embed_url = "https://app.powerbi.com/reportEmbed?reportId=f4f014c7-8869-4ab4-949c-e0256baf16c7&autoAuth=true&ctid=6f45f8df-317f-471f-80ad-4112734f0371"
    
    st.markdown("""
    ### Power BI Report
    
    View the embedded Power BI report below.
    """)
    
    # Embed the report using iframe
    st.markdown("#### 📈 Power BI Report:")
    
    # Create iframe HTML
    iframe_html = f"""
    <iframe 
        width="100%" 
        height="800" 
        src="{powerbi_embed_url}"
        frameborder="0" 
        allowFullScreen="true"
        style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    </iframe>
    """
    
    st.components.v1.html(iframe_html, height=820, scrolling=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sales Data Warehouse Analytics Platform | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

