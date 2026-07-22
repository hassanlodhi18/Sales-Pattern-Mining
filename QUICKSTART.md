# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Data Warehouse
Ensure you have the following files in `data/warehouse/`:
- `fact_sales.csv`
- `dim_customer.csv`
- `dim_date.csv`
- `dim_product.csv`
- `dim_store.csv`
- `dim_promotion.csv`

If missing, run the `data_warehousing.ipynb` notebook first.

### Step 3: Launch Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Available Features

### 1. EDA & Analysis
- Load and explore warehouse data
- View summary statistics
- Analyze sales by category, store, city, season, month

### 2. KDD Process
- Run complete KDD pipeline
- Configure sample size and mining tasks
- View insights and summary reports

### 3. Time Series Forecasting
- Forecast sales using ARIMA, SARIMA, ARMA
- Configure forecast steps and frequency
- Compare model performance

### 4. Association Rules
- Discover frequent patterns
- Generate association rules
- Analyze market basket insights

### 5. Visualizations
- Generate Matplotlib/Seaborn charts
- Create Plotly interactive dashboards
- Export visualizations

### 6. Power BI Integration
- Prepare data for Power BI
- Export aggregated tables
- Generate DAX measures

## 🔧 Command Line Usage

### Run KDD Process
```python
python -c "from kdd_process import KDDProcess; kdd = KDDProcess(); kdd.run_complete_process(sample_size=10000)"
```

### Run Forecasting
```python
python -c "from time_series_forecasting import TimeSeriesForecasting; f = TimeSeriesForecasting(); f.run_complete_forecasting()"
```

### Run Association Rules
```python
python -c "from association_rules_mining import AssociationRulesMining; arm = AssociationRulesMining(); arm.run_complete_analysis()"
```

### Generate Visualizations
```python
python -c "from visualizations import AdvancedVisualizations; viz = AdvancedVisualizations(); viz.create_all_visualizations()"
```

### Prepare Power BI Data
```python
python -c "from powerbi_integration import PowerBIIntegration; pbi = PowerBIIntegration(); pbi.prepare_all_for_powerbi()"
```

## 📊 Output Directories

After running analyses, check these directories:
- `kdd_results/` - KDD process outputs
- `forecast_output/` - Time series forecasts
- `association_rules_output/` - Association rules
- `visualizations/` - Charts and plots
- `powerbi_data/` - Power BI export files
- `eda_output/` - EDA results

## ⚠️ Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

If you see an error mentioning `statsmodels` specifically (e.g. "statsmodels not installed"), run:

```bash
pip install statsmodels
```

### Large Dataset Issues
Use sampling parameters:
- KDD: `sample_size=10000`
- Association Rules: `sample_size=50000`

### Memory Issues
Process in smaller chunks or use aggregated data.

## 📚 Next Steps

1. Explore the Streamlit GUI
2. Run EDA on your data
3. Generate forecasts
4. Discover association rules
5. Export to Power BI

Happy Analyzing! 🎉

