# How to View and Run the Project

## 🚀 Quick Start - View the Project

### Method 1: Interactive Web Application (Easiest - Recommended)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - The app will automatically open in your browser
   - If not, go to: `http://localhost:8501`
   - You'll see a navigation menu on the left sidebar

4. **Navigate Through Pages**
   - 🏠 **Home**: Overview and project information
   - 📈 **EDA & Analysis**: Explore your data
   - 🔄 **KDD Process**: Run knowledge discovery
   - 📉 **Time Series Forecasting**: Generate sales forecasts
   - 🛒 **Association Rules**: Discover market basket patterns
   - 📊 **Visualizations**: Create charts and dashboards
   - 🔌 **Power BI Integration**: Export data for Power BI

### Method 2: Command Line Interface

Run individual modules from command line:

```bash
# Run KDD Process
python kdd_process.py

# Run Time Series Forecasting
python time_series_forecasting.py

# Run Association Rules Mining
python association_rules_mining.py

# Generate Visualizations
python visualizations.py

# Prepare Power BI Data
python powerbi_integration.py

# Run EDA
python eda_warehouse.py
```

Or use the main entry point:

```bash
# Run all modules
python main.py all

# Run specific module
python main.py kdd
python main.py forecast
python main.py association
python main.py visualize
python main.py powerbi
python main.py eda
```

### Method 3: Jupyter Notebooks

You can also use the existing notebooks:
- `data_warehousing.ipynb` - Data warehouse creation
- `EDA.ipynb` - Exploratory data analysis
- `association_rule_mining.ipynb` - Association rules

## 📁 Viewing Output Files

After running analyses, check these output directories:

### 1. KDD Results
```
kdd_results/
├── insights.txt
├── summary_report.json
├── summary_statistics.csv
├── category_patterns.csv
└── monthly_patterns.csv
```

### 2. Forecast Outputs
```
forecast_output/
├── arima_forecast.png
├── sarima_forecast.png
├── arma_forecast.png
└── model_comparison.csv
```

### 3. Association Rules
```
association_rules_output/
├── frequent_itemsets.csv
├── association_rules.csv
└── insights.json
```

### 4. Visualizations
```
visualizations/
├── daily_sales_trend.png
├── sales_by_category.png
├── sales_by_store_type.png
├── monthly_category_heatmap.png
├── correlation_heatmap.png
├── interactive_sales_trend.html
├── interactive_dashboard.html
└── 3d_scatter.html
```

### 5. Power BI Data
```
powerbi_data/
├── fact_sales_powerbi.csv
├── daily_sales.csv
├── monthly_category_sales.csv
├── store_performance.csv
├── product_performance.csv
├── customer_segment.csv
├── powerbi_data.xlsx
├── powerbi_dax_measures.txt
└── powerbi_connection_guide.txt
```

### 6. EDA Outputs
```
eda_output/
├── sales_by_customer_category.png
├── sales_by_store_type.png
├── sales_by_city.png
├── sales_by_season.png
├── sales_by_month.png
├── sales_by_year.png
├── transaction_value_distribution.png
├── top_products_by_sales.png
└── correlation_heatmap.png
```

## 🖥️ System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Disk Space**: ~500MB for outputs
- **Browser**: Modern browser for Streamlit (Chrome, Firefox, Edge)

## 📋 Step-by-Step First Run

### Step 1: Verify Data Warehouse
```bash
# Check if warehouse files exist
ls data/warehouse/
```

You should see:
- `fact_sales.csv`
- `dim_customer.csv`
- `dim_date.csv`
- `dim_product.csv`
- `dim_store.csv`
- `dim_promotion.csv`

If missing, run `data_warehousing.ipynb` first.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Application
```bash
streamlit run app.py
```

### Step 4: Explore Features
1. Click "Load Data" in EDA & Analysis page
2. Explore different analysis options
3. Run KDD Process with sample data
4. Generate forecasts
5. Discover association rules
6. Create visualizations

## 🎯 Quick Test

To quickly test if everything works:

```bash
# Test imports
python -c "from kdd_process import KDDProcess; print('✓ KDD Process OK')"
python -c "from time_series_forecasting import TimeSeriesForecasting; print('✓ Forecasting OK')"
python -c "from association_rules_mining import AssociationRulesMining; print('✓ Association Rules OK')"
python -c "from visualizations import AdvancedVisualizations; print('✓ Visualizations OK')"
python -c "from powerbi_integration import PowerBIIntegration; print('✓ Power BI OK')"
python -c "import streamlit; print('✓ Streamlit OK')"
```

## 🔍 Viewing HTML Files

For interactive Plotly visualizations:
1. Navigate to `visualizations/` folder
2. Open `.html` files in your web browser
3. Interact with charts (zoom, pan, hover)

## 📊 Viewing Excel Files

For Power BI data:
1. Open `powerbi_data/powerbi_data.xlsx` in Excel
2. Each sheet contains different aggregated data
3. Use for Power BI import or manual analysis

## 🐛 Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Streamlit not opening
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

### Issue: Memory error
- Use sampling: Set `sample_size` parameter
- Process smaller datasets

### Issue: Data not loading
- Verify `data/warehouse/` files exist
- Check file paths in code

## 💡 Tips

1. **Start with Streamlit GUI** - Easiest way to explore
2. **Use sample sizes** - For faster testing (e.g., sample_size=10000)
3. **Check output folders** - All results are saved automatically
4. **Read documentation** - See README.md for detailed info
5. **Explore interactively** - Use Streamlit controls to adjust parameters

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. See `QUICKSTART.md` for quick reference
3. Review `PROJECT_SUMMARY.md` for overview
4. Check output directories for results

---

**Happy Analyzing! 🎉**

