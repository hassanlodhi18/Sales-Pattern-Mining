"""
Time Series Forecasting Module
Implements ARIMA, SARIMA, and ARMA models for sales forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import importlib
import subprocess
import sys


def ensure_statsmodels_installed(auto_install=False):
    """Check whether `statsmodels` is available and optionally install it.

    Returns True if available (or successfully installed), False otherwise.
    """
    try:
        importlib.import_module('statsmodels')
        return True
    except ImportError:
        msg = "statsmodels not installed. Install with: pip install statsmodels"
        if auto_install:
            print("statsmodels not found — attempting to install via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
                importlib.invalidate_caches()
                importlib.import_module('statsmodels')
                print("statsmodels installed successfully")
                return True
            except Exception as e:
                print(f"Automatic install failed: {e}")
                print(msg)
                return False
        else:
            print(msg)
            return False

# try:
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# except ImportError:
    # Provide a clearer message and instructions to install
    # print("Warning: statsmodels not installed. Please install: pip install statsmodels")
    # ARIMA = None
    # SARIMAX = None
    # ARMA = None

class TimeSeriesForecasting:
    """
    Time Series Forecasting using ARIMA, SARIMA, and ARMA models
    """
    
    def __init__(self, data=None, date_col='Date', value_col='Total_Cost', 
                 warehouse_path=None, start_date=None, end_date=None):
        """
        Initialize Time Series Forecasting
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input data. If None, will load from warehouse
        date_col : str
            Name of date column
        value_col : str
            Name of value column to forecast
        warehouse_path : str, optional
            Path to warehouse directory
        start_date : str or pd.Timestamp, optional
            Start date for filtering data
        end_date : str or pd.Timestamp, optional
            End date for filtering data
        """
        self.warehouse_path = warehouse_path or Path("data/warehouse")
        self.date_col = date_col
        self.value_col = value_col
        self.data = data
        self.time_series = None
        self.models = {}
        self.forecasts = {}
        self.model_metrics = {}
        
        if self.data is None:
            self._load_data(start_date=start_date, end_date=end_date)
    
    def _load_data(self, start_date=None, end_date=None):
        """
        Load data from warehouse
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp, optional
            Start date for filtering data
        end_date : str or pd.Timestamp, optional
            End date for filtering data
        """
        print("Loading data from warehouse...")
        try:
            fact = pd.read_csv(self.warehouse_path / "fact_sales.csv")
            dim_date = pd.read_csv(self.warehouse_path / "dim_date.csv")
            
            # Merge to get dates
            merged = fact.merge(dim_date, on='date_id', how='left')
            merged['Date'] = pd.to_datetime(merged['Date'])
            
            # Filter by date range if provided
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                merged = merged[merged['Date'] >= start_date]
                print(f"  Filtered: Start date >= {start_date.date()}")
            
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                merged = merged[merged['Date'] <= end_date]
                print(f"  Filtered: End date <= {end_date.date()}")
            
            # Aggregate daily sales
            daily_sales = merged.groupby(merged['Date'].dt.date)[self.value_col].sum().reset_index()
            daily_sales.columns = [self.date_col, self.value_col]
            daily_sales[self.date_col] = pd.to_datetime(daily_sales[self.date_col])
            daily_sales = daily_sales.sort_values(self.date_col)
            
            self.data = daily_sales
            print(f"[OK] Loaded {len(self.data)} daily records")
            if len(self.data) > 0:
                print(f"  Date range: {self.data[self.date_col].min().date()} to {self.data[self.date_col].max().date()}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_time_series(self, frequency='D', aggregation='sum'):
        """
        Prepare time series data
        
        Parameters:
        -----------
        frequency : str
            Time series frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        aggregation : str
            Aggregation method ('sum', 'mean', 'median')
        """
        print(f"\nPreparing time series (frequency: {frequency}, aggregation: {aggregation})...")
        
        if self.data is None:
            raise ValueError("No data available. Please load data first.")
        
        # Set date as index
        ts_data = self.data.set_index(self.date_col)[self.value_col]
        
        # Resample to desired frequency
        if aggregation == 'sum':
            ts_data = ts_data.resample(frequency).sum()
        elif aggregation == 'mean':
            ts_data = ts_data.resample(frequency).mean()
        elif aggregation == 'median':
            ts_data = ts_data.resample(frequency).median()
        
        # Remove NaN values
        ts_data = ts_data.dropna()
        
        self.time_series = ts_data
        print(f"[OK] Time series prepared: {len(ts_data)} periods")
        print(f"  Date range: {ts_data.index.min()} to {ts_data.index.max()}")
        print(f"  Mean value: ${ts_data.mean():.2f}")
        print(f"  Std deviation: ${ts_data.std():.2f}")
        
        return ts_data
    
    def check_stationarity(self, ts=None):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        ts : pd.Series, optional
            Time series to test. If None, uses self.time_series
        """
        if ts is None:
            ts = self.time_series
        
        if ts is None:
            raise ValueError("No time series data available")
        
        print("\nChecking stationarity...")
        result = adfuller(ts.dropna())
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        
        is_stationary = result[1] <= 0.05
        print(f"\n{'[OK] Series is stationary' if is_stationary else '[WARNING] Series is not stationary'}")
        
        return is_stationary, result
    
    def make_stationary(self, method='diff', diff_periods=1):
        """
        Make time series stationary
        
        Parameters:
        -----------
        method : str
            Method to use ('diff' for differencing, 'log' for log transform)
        diff_periods : int
            Number of periods for differencing
        """
        print(f"\nMaking series stationary using {method}...")
        
        if self.time_series is None:
            raise ValueError("No time series data available")
        
        if method == 'diff':
            self.time_series_stationary = self.time_series.diff(periods=diff_periods).dropna()
        elif method == 'log':
            self.time_series_stationary = np.log(self.time_series).dropna()
        elif method == 'log_diff':
            self.time_series_stationary = np.log(self.time_series).diff(periods=diff_periods).dropna()
        
        print(f"[OK] Stationary series created: {len(self.time_series_stationary)} periods")
        return self.time_series_stationary
    
    def find_optimal_arima_params(self, max_p=3, max_d=2, max_q=3, seasonal=False):
        """
        Find optimal ARIMA parameters using grid search
        
        Parameters:
        -----------
        max_p, max_d, max_q : int
            Maximum values for p, d, q parameters
        seasonal : bool
            Whether to use seasonal parameters
        """
        if ARIMA is None:
            print("statsmodels not available. Skipping parameter optimization.")
            return None
        
        print(f"\nFinding optimal ARIMA parameters (max_p={max_p}, max_d={max_d}, max_q={max_q})...")
        
        ts = self.time_series_stationary if hasattr(self, 'time_series_stationary') else self.time_series
        
        best_aic = np.inf
        best_params = None
        results = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        if seasonal:
                            model = SARIMAX(ts, order=(p, d, q), 
                                           seasonal_order=(1, 1, 1, 12))
                        else:
                            model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        results.append({'p': p, 'd': d, 'q': q, 'AIC': aic})
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        print(f"[OK] Best parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params, results
    
    def fit_arima(self, order=(1, 1, 1), ts=None):
        """
        Fit ARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) parameters for ARIMA
        ts : pd.Series, optional
            Time series to fit. If None, uses self.time_series
        """
        if ARIMA is None:
            raise ImportError("statsmodels is required. Install: pip install statsmodels")
        
        print(f"\nFitting ARIMA{order} model...")
        
        if ts is None:
            ts = self.time_series
        
        if ts is None:
            raise ValueError("No time series data available")
        
        try:
            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            
            self.models['ARIMA'] = fitted_model
            self.model_metrics['ARIMA'] = {
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'order': order
            }
            
            print(f"[OK] ARIMA{order} model fitted")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  BIC: {fitted_model.bic:.2f}")
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    
    def fit_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), ts=None):
        """
        Fit SARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) parameters
        seasonal_order : tuple
            (P, D, Q, s) seasonal parameters
        ts : pd.Series, optional
            Time series to fit
        """
        if SARIMAX is None:
            raise ImportError("statsmodels is required. Install: pip install statsmodels")
        
        print(f"\nFitting SARIMA{order}x{seasonal_order} model...")
        
        if ts is None:
            ts = self.time_series
        
        if ts is None:
            raise ValueError("No time series data available")
        
        try:
            model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            self.models['SARIMA'] = fitted_model
            self.model_metrics['SARIMA'] = {
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'order': order,
                'seasonal_order': seasonal_order
            }
            
            print(f"[OK] SARIMA{order}x{seasonal_order} model fitted")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  BIC: {fitted_model.bic:.2f}")
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting SARIMA model: {e}")
            return None
    
    def fit_arma(self, order=(1, 1), ts=None):
        """
        Fit ARMA model (assumes stationary data)
        ARMA is equivalent to ARIMA with d=0
        
        Parameters:
        -----------
        order : tuple
            (p, q) parameters for ARMA
        ts : pd.Series, optional
            Time series to fit
        """
        if ARIMA is None:
            raise ImportError("statsmodels is required. Install: pip install statsmodels")
        
        print(f"\nFitting ARMA{order} model...")
        
        if ts is None:
            ts = self.time_series_stationary if hasattr(self, 'time_series_stationary') else self.time_series
        
        if ts is None:
            raise ValueError("No time series data available")
        
        try:
            # ARMA(p, q) is equivalent to ARIMA(p, 0, q)
            arima_order = (order[0], 0, order[1])
            model = ARIMA(ts, order=arima_order)
            fitted_model = model.fit()
            
            self.models['ARMA'] = fitted_model
            self.model_metrics['ARMA'] = {
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'order': order
            }
            
            print(f"[OK] ARMA{order} model fitted")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  BIC: {fitted_model.bic:.2f}")
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARMA model: {e}")
            return None
    
    def forecast(self, model_name='ARIMA', steps=30, plot=True, output_dir='forecast_output'):
        """
        Generate forecasts using fitted model
        
        Parameters:
        -----------
        model_name : str
            Name of model to use ('ARIMA', 'SARIMA', 'ARMA')
        steps : int
            Number of periods to forecast
        plot : bool
            Whether to create forecast plots
        output_dir : str
            Directory to save results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not fitted. Please fit the model first.")
        
        print(f"\nGenerating {steps}-period forecast using {model_name}...")
        
        model = self.models[model_name]
        forecast_result = model.forecast(steps=steps)
        forecast_ci = model.get_forecast(steps=steps).conf_int()
        
        self.forecasts[model_name] = {
            'forecast': forecast_result,
            'confidence_intervals': forecast_ci,
            'steps': steps
        }
        
        print(f"[OK] Forecast generated")
        print(f"  Forecast mean: ${forecast_result.mean():.2f}")
        print(f"  Forecast range: ${forecast_result.min():.2f} - ${forecast_result.max():.2f}")
        
        if plot:
            self.plot_forecast(model_name, output_dir)
        
        return forecast_result, forecast_ci
    
    def plot_forecast(self, model_name='ARIMA', output_dir='forecast_output'):
        """Plot forecast results"""
        if model_name not in self.forecasts:
            raise ValueError(f"No forecast available for {model_name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        forecast_data = self.forecasts[model_name]
        forecast = forecast_data['forecast']
        ci = forecast_data['confidence_intervals']
        
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        if self.time_series is not None:
            plt.plot(self.time_series.index[-100:], self.time_series.values[-100:], 
                    label='Historical Data', color='blue', linewidth=2)
        
        # Plot forecast
        forecast_index = pd.date_range(start=self.time_series.index[-1] + pd.Timedelta(days=1), 
                                       periods=len(forecast), freq='D')
        plt.plot(forecast_index, forecast, label=f'{model_name} Forecast', 
                color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        plt.fill_between(forecast_index, ci.iloc[:, 0], ci.iloc[:, 1], 
                        alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.title(f'Sales Forecast using {model_name} Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_path / f'{model_name.lower()}_forecast.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved forecast plot: {filename}")
        plt.close()
    
    def compare_models(self, output_dir='forecast_output'):
        """Compare all fitted models"""
        if not self.model_metrics:
            print("No models fitted yet.")
            return
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(self.model_metrics).T
        print("\nModel Metrics:")
        print(comparison_df)
        
        # Save comparison
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        comparison_df.to_csv(output_path / 'model_comparison.csv')
        print(f"\n[OK] Saved model comparison to {output_path / 'model_comparison.csv'}")
        
        return comparison_df
    
    def run_complete_forecasting(self, forecast_steps=30, output_dir='forecast_output'):
        """
        Run complete forecasting pipeline
        
        Parameters:
        -----------
        forecast_steps : int
            Number of periods to forecast
        output_dir : str
            Directory to save results
        """
        print("\n" + "=" * 80)
        print("COMPLETE TIME SERIES FORECASTING PIPELINE")
        print("=" * 80)
        
        # Prepare time series
        self.prepare_time_series(frequency='D', aggregation='sum')
        
        # Check stationarity
        is_stationary, _ = self.check_stationarity()
        
        # Make stationary if needed
        if not is_stationary:
            self.make_stationary(method='diff', diff_periods=1)
            ts_for_modeling = self.time_series_stationary
        else:
            ts_for_modeling = self.time_series
        
        # Fit models
        print("\n" + "-" * 80)
        print("FITTING MODELS")
        print("-" * 80)
        
        # ARIMA
        self.fit_arima(order=(1, 1, 1), ts=self.time_series)
        
        # SARIMA (with seasonal component)
        try:
            self.fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), ts=self.time_series)
        except:
            print("Could not fit SARIMA model")
        
        # ARMA (on stationary data)
        if hasattr(self, 'time_series_stationary'):
            try:
                self.fit_arma(order=(1, 1), ts=self.time_series_stationary)
            except:
                print("Could not fit ARMA model")
        
        # Compare models
        self.compare_models(output_dir)
        
        # Generate forecasts
        print("\n" + "-" * 80)
        print("GENERATING FORECASTS")
        print("-" * 80)
        
        for model_name in self.models.keys():
            try:
                self.forecast(model_name=model_name, steps=forecast_steps, 
                            plot=True, output_dir=output_dir)
            except Exception as e:
                print(f"Error forecasting with {model_name}: {e}")
        
        print("\n" + "=" * 80)
        print("FORECASTING PIPELINE COMPLETE!")
        print("=" * 80)
        
        return {
            'models': self.models,
            'forecasts': self.forecasts,
            'metrics': self.model_metrics
        }


# Example usage
if __name__ == "__main__":
    # Initialize forecasting
    forecaster = TimeSeriesForecasting()
    
    # Run complete forecasting pipeline
    results = forecaster.run_complete_forecasting(forecast_steps=30)
    
    print("\nForecasting completed successfully!")

