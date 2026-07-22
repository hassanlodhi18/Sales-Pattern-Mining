"""
KDD (Knowledge Discovery in Databases) Process Implementation
Implements all 5 steps: Selection, Preprocessing, Transformation, Data Mining, Interpretation
"""

import pandas as pd
import numpy as np 
import statsmodels
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KDDProcess:
    """
    Complete KDD Process Implementation
    """
    
    def __init__(self, raw_data_path=None, warehouse_path=None):
        """
        Initialize KDD Process
        
        Parameters:
        -----------
        raw_data_path : str, optional
            Path to raw data file
        warehouse_path : str, optional
            Path to warehouse directory
        """
        self.raw_data_path = raw_data_path or "data/raw/Retail_Transactions_Dataset.csv"
        self.warehouse_path = warehouse_path or Path("data/warehouse")
        self.raw_data = None
        self.processed_data = None
        self.transformed_data = None
        self.mining_results = {}
        
    # ==================== STEP 1: SELECTION ====================
    def selection(self, sample_size=None, date_range=None, columns=None):
        """
        Step 1: Data Selection
        Select relevant data from the dataset
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of records to select (for testing)
        date_range : tuple, optional
            (start_date, end_date) to filter by date
        columns : list, optional
            Specific columns to select
        """
        print("=" * 80)
        print("STEP 1: DATA SELECTION")
        print("=" * 80)
        
        # Load raw data
        print(f"Loading data from: {self.raw_data_path}")
        self.raw_data = pd.read_csv(self.raw_data_path)
        print(f"Original data shape: {self.raw_data.shape}")
        print(f"Original columns: {list(self.raw_data.columns)}")
        
        # Apply filters
        if date_range:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
            self.raw_data = self.raw_data[
                (self.raw_data['Date'] >= date_range[0]) & 
                (self.raw_data['Date'] <= date_range[1])
            ]
            print(f"After date filtering: {self.raw_data.shape}")
        
        if columns:
            self.raw_data = self.raw_data[columns]
            print(f"After column selection: {self.raw_data.shape}")
        
        if sample_size:
            self.raw_data = self.raw_data.sample(n=min(sample_size, len(self.raw_data)), 
                                                  random_state=42)
            print(f"After sampling: {self.raw_data.shape}")
        
        print(f"[OK] Selection complete. Selected {len(self.raw_data):,} records")
        return self.raw_data
    
    # ==================== STEP 2: PREPROCESSING ====================
    def preprocessing(self):
        """
        Step 2: Data Preprocessing
        Clean and prepare data for analysis
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 80)
        
        if self.raw_data is None:
            raise ValueError("Please run selection() first")
        
        self.processed_data = self.raw_data.copy()
        
        # 1. Handle missing values
        print("\n1. Handling Missing Values:")
        missing_before = self.processed_data.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        
        # Fill missing promotions with 'None'
        if 'Promotion' in self.processed_data.columns:
            self.processed_data['Promotion'] = self.processed_data['Promotion'].fillna('None')
        
        # Fill other missing values
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        self.processed_data[numeric_cols] = self.processed_data[numeric_cols].fillna(
            self.processed_data[numeric_cols].median()
        )
        
        missing_after = self.processed_data.isnull().sum().sum()
        print(f"   Missing values after: {missing_after}")
        
        # 2. Handle duplicates
        print("\n2. Handling Duplicates:")
        duplicates_before = self.processed_data.duplicated().sum()
        print(f"   Duplicate records: {duplicates_before}")
        self.processed_data = self.processed_data.drop_duplicates()
        print(f"   Records after removing duplicates: {len(self.processed_data):,}")
        
        # 3. Data type conversion
        print("\n3. Data Type Conversion:")
        if 'Date' in self.processed_data.columns:
            self.processed_data['Date'] = pd.to_datetime(self.processed_data['Date'])
            print("   [OK] Converted Date to datetime")
        
        # Convert Product column from string to list
        if 'Product' in self.processed_data.columns:
            import ast
            try:
                self.processed_data['Product'] = self.processed_data['Product'].apply(ast.literal_eval)
                print("   [OK] Converted Product to list")
            except:
                print("   ⚠ Could not convert Product column")
        
        # 4. Handle outliers (using IQR method for numeric columns)
        print("\n4. Handling Outliers:")
        for col in ['Total_Cost', 'Total_Items']:
            if col in self.processed_data.columns:
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.processed_data[col] < lower_bound) | 
                                  (self.processed_data[col] > upper_bound)).sum()
                
                # Cap outliers instead of removing (to preserve data volume)
                self.processed_data[col] = self.processed_data[col].clip(
                    lower=lower_bound, upper=upper_bound
                )
                
                print(f"   {col}: Capped {outliers_before} outliers")
        
        # 5. Data validation
        print("\n5. Data Validation:")
        print(f"   [OK] Total records: {len(self.processed_data):,}")
        print(f"   [OK] Date range: {self.processed_data['Date'].min()} to {self.processed_data['Date'].max()}")
        print(f"   [OK] Unique customers: {self.processed_data['Customer_Name'].nunique():,}")
        print(f"   [OK] Unique transactions: {self.processed_data['Transaction_ID'].nunique():,}")
        
        print(f"\n[OK] Preprocessing complete. Processed {len(self.processed_data):,} records")
        return self.processed_data
    
    # ==================== STEP 3: TRANSFORMATION ====================
    def transformation(self):
        """
        Step 3: Data Transformation
        Transform data into suitable format for data mining
        """
        print("\n" + "=" * 80)
        print("STEP 3: DATA TRANSFORMATION")
        print("=" * 80)
        
        if self.processed_data is None:
            raise ValueError("Please run preprocessing() first")
        
        self.transformed_data = self.processed_data.copy()
        
        # 1. Feature engineering
        print("\n1. Feature Engineering:")
        
        # Extract temporal features
        if 'Date' in self.transformed_data.columns:
            self.transformed_data['year'] = self.transformed_data['Date'].dt.year
            self.transformed_data['month'] = self.transformed_data['Date'].dt.month
            self.transformed_data['day'] = self.transformed_data['Date'].dt.day
            self.transformed_data['weekday'] = self.transformed_data['Date'].dt.day_name()
            self.transformed_data['quarter'] = self.transformed_data['Date'].dt.quarter
            self.transformed_data['is_weekend'] = self.transformed_data['Date'].dt.dayofweek >= 5
            print("   [OK] Extracted temporal features")
        
        # Calculate derived metrics
        if 'Total_Cost' in self.transformed_data.columns and 'Total_Items' in self.transformed_data.columns:
            self.transformed_data['avg_item_price'] = (
                self.transformed_data['Total_Cost'] / self.transformed_data['Total_Items']
            )
            print("   [OK] Calculated average item price")
        
        # 2. Encode categorical variables
        print("\n2. Categorical Encoding:")
        
        # Label encoding for categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['Payment_Method', 'City', 'Store_Type', 
                           'Customer_Category', 'Season', 'Promotion']
        
        self.label_encoders = {}
        for col in categorical_cols:
            if col in self.transformed_data.columns:
                le = LabelEncoder()
                self.transformed_data[f'{col}_encoded'] = le.fit_transform(
                    self.transformed_data[col].astype(str)
                )
                self.label_encoders[col] = le
                print(f"   [OK] Encoded {col}")
        
        # 3. Create aggregated features
        print("\n3. Creating Aggregated Features:")
        
        # Customer-level aggregations
        if 'Customer_Name' in self.transformed_data.columns:
            customer_stats = self.transformed_data.groupby('Customer_Name').agg({
                'Total_Cost': ['sum', 'mean', 'count'],
                'Total_Items': 'sum'
            }).reset_index()
            customer_stats.columns = ['Customer_Name', 'customer_total_spent', 
                                     'customer_avg_transaction', 'customer_transaction_count',
                                     'customer_total_items']
            self.transformed_data = self.transformed_data.merge(customer_stats, 
                                                               on='Customer_Name', 
                                                               how='left')
            print("   [OK] Created customer-level aggregations")
        
        # 4. Normalization (for numeric features)
        print("\n4. Normalization:")
        from sklearn.preprocessing import StandardScaler
        
        numeric_features = ['Total_Cost', 'Total_Items', 'avg_item_price']
        if 'avg_item_price' in self.transformed_data.columns:
            scaler = StandardScaler()
            for col in numeric_features:
                if col in self.transformed_data.columns:
                    self.transformed_data[f'{col}_normalized'] = scaler.fit_transform(
                        self.transformed_data[[col]]
                    )
            print("   [OK] Normalized numeric features")
        
        # 5. Create time series aggregation
        print("\n5. Time Series Aggregation:")
        if 'Date' in self.transformed_data.columns:
            # Daily sales aggregation
            daily_sales = self.transformed_data.groupby(
                self.transformed_data['Date'].dt.date
            )['Total_Cost'].agg(['sum', 'mean', 'count']).reset_index()
            daily_sales.columns = ['date', 'daily_total_sales', 'daily_avg_sales', 'daily_transaction_count']
            self.daily_sales = daily_sales
            print(f"   [OK] Created daily sales aggregation ({len(daily_sales)} days)")
        
        print(f"\n[OK] Transformation complete. Transformed {len(self.transformed_data):,} records")
        return self.transformed_data
    
    # ==================== STEP 4: DATA MINING ====================
    def data_mining(self, mining_tasks=None):
        """
        Step 4: Data Mining
        Apply various data mining techniques
        
        Parameters:
        -----------
        mining_tasks : list, optional
            List of mining tasks to perform: ['summary', 'patterns', 'clustering', 'correlation']
        """
        print("\n" + "=" * 80)
        print("STEP 4: DATA MINING")
        print("=" * 80)
        
        if self.transformed_data is None:
            raise ValueError("Please run transformation() first")
        
        if mining_tasks is None:
            mining_tasks = ['summary', 'patterns', 'correlation']
        
        # 1. Summary Statistics
        if 'summary' in mining_tasks:
            print("\n1. Summary Statistics Mining:")
            summary_stats = self.transformed_data.describe()
            self.mining_results['summary_statistics'] = summary_stats
            print("   [OK] Generated summary statistics")
            print(summary_stats)
        
        # 2. Pattern Discovery
        if 'patterns' in mining_tasks:
            print("\n2. Pattern Discovery:")
            
            # Sales patterns by category
            if 'Customer_Category' in self.transformed_data.columns:
                category_patterns = self.transformed_data.groupby('Customer_Category').agg({
                    'Total_Cost': ['sum', 'mean', 'count'],
                    'Total_Items': 'mean'
                })
                self.mining_results['category_patterns'] = category_patterns
                print("   [OK] Discovered customer category patterns")
            
            # Temporal patterns
            if 'month' in self.transformed_data.columns:
                monthly_patterns = self.transformed_data.groupby('month').agg({
                    'Total_Cost': ['sum', 'mean'],
                    'Transaction_ID': 'count'
                })
                self.mining_results['monthly_patterns'] = monthly_patterns
                print("   [OK] Discovered monthly patterns")
            
            # Store patterns
            if 'Store_Type' in self.transformed_data.columns:
                store_patterns = self.transformed_data.groupby('Store_Type').agg({
                    'Total_Cost': ['sum', 'mean', 'count']
                })
                self.mining_results['store_patterns'] = store_patterns
                print("   [OK] Discovered store type patterns")
        
        # 3. Correlation Analysis
        if 'correlation' in mining_tasks:
            print("\n3. Correlation Analysis:")
            numeric_cols = self.transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = self.transformed_data[numeric_cols].corr()
                self.mining_results['correlation_matrix'] = correlation_matrix
                print("   [OK] Generated correlation matrix")
                print(f"   Strong correlations (|r| > 0.5):")
                high_corr = correlation_matrix[
                    (correlation_matrix.abs() > 0.5) & 
                    (correlation_matrix != 1.0)
                ].stack()
                if len(high_corr) > 0:
                    print(high_corr)
        
        # 4. Clustering (optional)
        if 'clustering' in mining_tasks:
            print("\n4. Clustering Analysis:")
            from sklearn.cluster import KMeans
            
            # Select features for clustering
            cluster_features = ['Total_Cost', 'Total_Items']
            if 'avg_item_price' in self.transformed_data.columns:
                cluster_features.append('avg_item_price')
            
            available_features = [f for f in cluster_features if f in self.transformed_data.columns]
            
            if len(available_features) >= 2:
                X = self.transformed_data[available_features].fillna(0)
                
                # Standardize
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                self.transformed_data['cluster'] = kmeans.fit_predict(X_scaled)
                self.mining_results['clustering'] = {
                    'model': kmeans,
                    'n_clusters': 3,
                    'cluster_sizes': pd.Series(self.transformed_data['cluster']).value_counts().to_dict()
                }
                print("   [OK] Performed KMeans clustering (3 clusters)")
                print(f"   Cluster sizes: {self.mining_results['clustering']['cluster_sizes']}")
        
        print(f"\n[OK] Data mining complete. Generated {len(self.mining_results)} result sets")
        return self.mining_results
    
    # ==================== STEP 5: INTERPRETATION ====================
    def interpretation(self, output_path="kdd_results"):
        """
        Step 5: Interpretation
        Interpret and present the mining results
        
        Parameters:
        -----------
        output_path : str
            Path to save interpretation results
        """
        print("\n" + "=" * 80)
        print("STEP 5: INTERPRETATION")
        print("=" * 80)
        
        if not self.mining_results:
            raise ValueError("Please run data_mining() first")
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Generate insights report
        print("\n1. Generating Insights Report:")
        insights = []
        
        # Summary insights
        if 'summary_statistics' in self.mining_results:
            stats = self.mining_results['summary_statistics']
            if 'Total_Cost' in stats.columns:
                insights.append(f"Average transaction value: ${stats.loc['mean', 'Total_Cost']:.2f}")
                insights.append(f"Median transaction value: ${stats.loc['50%', 'Total_Cost']:.2f}")
                insights.append(f"Maximum transaction value: ${stats.loc['max', 'Total_Cost']:.2f}")
        
        # Pattern insights
        if 'category_patterns' in self.mining_results:
            patterns = self.mining_results['category_patterns']
            top_category = patterns[('Total_Cost', 'sum')].idxmax()
            insights.append(f"Highest spending category: {top_category}")
        
        if 'monthly_patterns' in self.mining_results:
            patterns = self.mining_results['monthly_patterns']
            best_month = patterns[('Total_Cost', 'sum')].idxmax()
            insights.append(f"Best sales month: {best_month}")
        
        # Save insights
        with open(output_dir / "insights.txt", "w") as f:
            f.write("KDD PROCESS INTERPRETATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
                print(f"   {i}. {insight}")
        
        # 2. Save mining results
        print("\n2. Saving Mining Results:")
        for key, value in self.mining_results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(output_dir / f"{key}.csv")
                print(f"   [OK] Saved {key}.csv")
        
        # 3. Generate summary report
        print("\n3. Generating Summary Report:")
        summary_report = {
            'total_records': len(self.transformed_data),
            'date_range': f"{self.transformed_data['Date'].min()} to {self.transformed_data['Date'].max()}",
            'mining_tasks_completed': list(self.mining_results.keys()),
            'insights_count': len(insights)
        }
        
        import json
        with open(output_dir / "summary_report.json", "w") as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print(f"   [OK] Saved summary_report.json")
        print(f"\n[OK] Interpretation complete. Results saved to '{output_dir}'")
        
        return insights, summary_report
    
    # ==================== RUN COMPLETE KDD PROCESS ====================
    def run_complete_process(self, sample_size=None, date_range=None, 
                           mining_tasks=None, output_path="kdd_results"):
        """
        Run the complete KDD process from start to finish
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of records to process
        date_range : tuple, optional
            (start_date, end_date) for filtering
        mining_tasks : list, optional
            Mining tasks to perform
        output_path : str
            Path to save results
        """
        print("\n" + "=" * 80)
        print("COMPLETE KDD PROCESS EXECUTION")
        print("=" * 80)
        
        # Step 1: Selection
        self.selection(sample_size=sample_size, date_range=date_range)
        
        # Step 2: Preprocessing
        self.preprocessing()
        
        # Step 3: Transformation
        self.transformation()
        
        # Step 4: Data Mining
        self.data_mining(mining_tasks=mining_tasks)
        
        # Step 5: Interpretation
        insights, summary = self.interpretation(output_path=output_path)
        
        print("\n" + "=" * 80)
        print("KDD PROCESS COMPLETE!")
        print("=" * 80)
        
        return {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'transformed_data': self.transformed_data,
            'mining_results': self.mining_results,
            'insights': insights,
            'summary': summary
        }


# Example usage
if __name__ == "__main__":
    # Initialize KDD Process
    kdd = KDDProcess()
    
    # Run complete process (using sample for faster execution)
    results = kdd.run_complete_process(
        sample_size=10000,  # Use sample for testing
        mining_tasks=['summary', 'patterns', 'correlation']
    )
    
    print("\nKDD Process completed successfully!")

