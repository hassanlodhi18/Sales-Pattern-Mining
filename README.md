# Sales Data Warehouse Analytics Platform

A comprehensive analytics platform for sales data mining, forecasting, and visualization.

## 🎯 Objectives

1. **Large-scale Sales Data Warehouse** - Star schema data warehouse with fact and dimension tables
2. **KDD Process Implementation** - Complete Knowledge Discovery in Databases pipeline
3. **Time Series Forecasting** - ARIMA, SARIMA, and ARMA models for sales prediction
4. **Association Rules Mining** - Frequent patterns and market basket analysis
5. **Advanced Visualizations** - Matplotlib, Seaborn, and Plotly interactive charts
6. **Power BI Integration** - Data export and DAX measures for Power BI dashboards
7. **Interactive GUI** - Streamlit-based web application

## 📁 Project Structure

```
Sales_pattern_mining/
├── data/
│   ├── raw/
│   │   └── Retail_Transactions_Dataset.csv
│   └── warehouse/
│       ├── fact_sales.csv
│       ├── dim_customer.csv
│       ├── dim_date.csv
│       ├── dim_product.csv
│       ├── dim_store.csv
│       └── dim_promotion.csv
├── kdd_process.py              # KDD process implementation
├── time_series_forecasting.py   # ARIMA, SARIMA, ARMA models
├── association_rules_mining.py  # Market basket analysis
├── visualizations.py            # Advanced visualizations
├── powerbi_integration.py       # Power BI data export
├── eda_warehouse.py             # Exploratory data analysis
├── app.py                       # Streamlit GUI application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

1. **Clone or download the project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data warehouse structure:**
   - Ensure `data/warehouse/` contains all dimension and fact tables
   - If not, run the data warehousing notebook to create them

## 💻 Usage

### Option 1: Interactive Web Application (Recommended)

Launch the Streamlit GUI:
```bash
streamlit run app.py
```

This opens a web interface with:
- 📈 EDA & Analysis
- 🔄 KDD Process
- 📉 Time Series Forecasting
- 🛒 Association Rules
- 📊 Visualizations
- 🔌 Power BI Integration

### Option 2: Command Line Scripts

#### Run KDD Process
```python
from kdd_process import KDDProcess

kdd = KDDProcess()
results = kdd.run_complete_process(
    sample_size=10000,
    mining_tasks=['summary', 'patterns', 'correlation']
)
```

#### Run Time Series Forecasting
```python
from time_series_forecasting import TimeSeriesForecasting

forecaster = TimeSeriesForecasting()
results = forecaster.run_complete_forecasting(forecast_steps=30)
```

#### Run Association Rules Mining
```python
from association_rules_mining import AssociationRulesMining

arm = AssociationRulesMining()
results = arm.run_complete_analysis(
    min_support=0.02,
    min_confidence=0.5
)
```

#### Generate Visualizations
```python
from visualizations import AdvancedVisualizations

viz = AdvancedVisualizations()
viz.create_all_visualizations()
```

#### Prepare Power BI Data
```python
from powerbi_integration import PowerBIIntegration

pbi = PowerBIIntegration()
pbi.prepare_all_for_powerbi()
```

## 📊 Features

### 1. KDD Process
- **Selection**: Data filtering and sampling
- **Preprocessing**: Missing value handling, outlier treatment, data cleaning
- **Transformation**: Feature engineering, encoding, normalization
- **Data Mining**: Pattern discovery, clustering, correlation analysis
- **Interpretation**: Insights generation and reporting

### 2. Time Series Forecasting
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA
- **ARMA**: AutoRegressive Moving Average
- Automatic stationarity checking
- Model comparison and evaluation

### 3. Association Rules Mining
- **Apriori Algorithm**: Frequent itemset mining
- **FP-Growth**: Alternative algorithm
- **Association Rules**: Confidence, support, lift metrics
- Market basket analysis insights

### 4. Visualizations
- **Matplotlib**: Static charts and plots
- **Seaborn**: Statistical visualizations and heatmaps
- **Plotly**: Interactive dashboards and 3D plots

### 5. Power BI Integration
- Fact table preparation
- Aggregated tables for performance
- Excel export
- DAX measures generation
- Connection guide

## 📈 Data Warehouse Schema

### Fact Table
- `fact_sales.csv`: Transaction-level sales data

### Dimension Tables
- `dim_customer.csv`: Customer information
- `dim_date.csv`: Date attributes and hierarchies
- `dim_product.csv`: Product catalog
- `dim_store.csv`: Store locations and types
- `dim_promotion.csv`: Promotion details

## 🔧 Configuration

### KDD Process Parameters
- `sample_size`: Number of records to process
- `mining_tasks`: List of mining techniques to apply

### Forecasting Parameters
- `forecast_steps`: Number of periods to forecast
- `frequency`: Time series frequency (D/W/M)
- `aggregation`: Aggregation method (sum/mean/median)

### Association Rules Parameters
- `min_support`: Minimum support threshold
- `min_confidence`: Minimum confidence threshold
- `algorithm`: 'apriori' or 'fpgrowth'

## 📝 Output Directories

- `kdd_results/`: KDD process outputs
- `forecast_output/`: Time series forecasts and plots
- `association_rules_output/`: Frequent itemsets and rules
- `visualizations/`: Generated charts and plots
- `powerbi_data/`: Power BI export files
- `eda_output/`: EDA analysis results

## 🛠️ Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Large Dataset Issues
For very large datasets, use sampling:
- KDD Process: Set `sample_size` parameter
- Association Rules: Use `sample_size` in `load_transactions()`

### Memory Issues
- Process data in chunks
- Use aggregated tables for analysis
- Sample data for testing

## 📚 Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib/seaborn: Static visualizations
- plotly: Interactive visualizations
- scikit-learn: Machine learning utilities
- mlxtend: Association rules mining
- statsmodels: Time series forecasting
- streamlit: Web application framework
- openpyxl: Excel file support

## 🤝 Contributing

This is a comprehensive analytics platform. To extend:

1. Add new mining algorithms in respective modules
2. Create new visualization types in `visualizations.py`
3. Extend Power BI integration with additional measures
4. Add new pages to Streamlit app

## 📄 License

This project is for educational and analytical purposes.

## 🎓 Academic Use

This platform implements:
- KDD (Knowledge Discovery in Databases) process
- Time series forecasting models (ARIMA, SARIMA, ARMA)
- Association rule mining (Apriori, FP-Growth)
- Data warehouse design (Star schema)
- Business intelligence integration

---

**Built with ❤️ for Sales Pattern Mining and Analytics**

