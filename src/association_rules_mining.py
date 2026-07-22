"""
Enhanced Association Rules Mining Module
Generates Frequent Patterns, Association Rules, and Data Insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Fix Windows console encoding for Unicode characters
# Note: This redirection is skipped in web/interactive environments like Streamlit
# to avoid "I/O operation on closed file" errors
if sys.platform == 'win32':
    # Only attempt redirection in standard console environments
    # Check if we're likely in a web/streamlit environment
    is_web_env = (
        any(module in sys.modules for module in ['streamlit', 'flask', 'django', 'tornado']) or
        'streamlit' in str(type(sys.stdout)).lower() or
        'streamlit' in str(type(sys.stderr)).lower()
    )
    
    if not is_web_env:
        try:
            # Only redirect if stdout has buffer attribute, is not already wrapped, and buffer is not closed
            if (hasattr(sys.stdout, 'buffer') and 
                not isinstance(sys.stdout, io.TextIOWrapper)):
                buffer = sys.stdout.buffer
                # Check if buffer is closed if it has that attribute
                if not (hasattr(buffer, 'closed') and buffer.closed):
                    sys.stdout = io.TextIOWrapper(buffer, encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError, io.UnsupportedOperation):
            pass  # Skip redirection if not possible
        
        try:
            # Same for stderr
            if (hasattr(sys.stderr, 'buffer') and 
                not isinstance(sys.stderr, io.TextIOWrapper)):
                buffer = sys.stderr.buffer
                if not (hasattr(buffer, 'closed') and buffer.closed):
                    sys.stderr = io.TextIOWrapper(buffer, encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError, io.UnsupportedOperation):
            pass  # Skip redirection if not possible

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
except ImportError:
    print("Warning: mlxtend not installed. Please install: pip install mlxtend")
    TransactionEncoder = None
    apriori = None
    association_rules = None
    fpgrowth = None

class AssociationRulesMining:
    """
    Association Rules Mining for Market Basket Analysis
    """
    
    def __init__(self, warehouse_path=None):
        """
        Initialize Association Rules Mining
        
        Parameters:
        -----------
        warehouse_path : str, optional
            Path to warehouse directory
        """
        self.warehouse_path = warehouse_path or Path("data/warehouse")
        self.transactions = None
        self.transaction_matrix = None
        self.frequent_itemsets = None
        self.association_rules = None
        self.insights = {}
        
    def load_transactions(self, sample_size=None):
        """
        Load transactions from warehouse (optimized for speed)
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of transactions to sample
        """
        print("Loading transactions from warehouse...")
        
        try:
            # Load dimension table first (smaller)
            dim_product = pd.read_csv(self.warehouse_path / "dim_product.csv")
            
            # Create product mapping dictionary for faster lookup
            if 'product_key' in dim_product.columns:
                product_map = dict(zip(dim_product['product_key'], dim_product['Product']))
                merge_key = 'product_key'
            else:
                product_map = dict(zip(dim_product['product_id'], dim_product['Product']))
                merge_key = 'product_id'
            
            # Load fact table with optimized dtypes
            fact = pd.read_csv(
                self.warehouse_path / "fact_sales.csv",
                dtype={'product_id': 'int64', 'Transaction_ID': 'int64'},
                usecols=['Transaction_ID', 'product_id']  # Only load needed columns
            )
            
            # Sample transactions early if sample_size is provided (much faster)
            if sample_size and sample_size > 0:
                unique_transactions = fact['Transaction_ID'].unique()
                if len(unique_transactions) > sample_size:
                    import random
                    random.seed(42)
                    sampled_transaction_ids = set(random.sample(unique_transactions.tolist(), sample_size))
                    fact = fact[fact['Transaction_ID'].isin(sampled_transaction_ids)]
            
            # Map product IDs to product names using dictionary (faster than merge)
            fact['Product'] = fact['product_id'].map(product_map)
            
            # Remove rows with missing products
            fact = fact[fact['Product'].notna()]
            fact = fact[fact['Product'] != 'nan']
            
            # Group by transaction to get product lists (optimized)
            transactions = fact.groupby('Transaction_ID')['Product'].apply(
                lambda x: x.tolist()
            ).tolist()
            
            # Remove empty transactions
            transactions = [t for t in transactions if len(t) > 0]
            
            self.transactions = transactions
            print(f"[OK] Loaded {len(transactions):,} transactions")
            if len(transactions) > 0:
                print(f"  Average items per transaction: {np.mean([len(t) for t in transactions]):.2f}")
                print(f"  Max items in transaction: {max([len(t) for t in transactions])}")
            
            return transactions
        except Exception as e:
            print(f"Error loading transactions: {e}")
            raise
    
    def prepare_transaction_matrix(self, min_support=0.01):
        """
        Prepare transaction matrix using TransactionEncoder
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold
        """
        if TransactionEncoder is None:
            raise ImportError("mlxtend is required. Install: pip install mlxtend")
        
        if self.transactions is None:
            raise ValueError("No transactions loaded. Please run load_transactions() first.")
        
        print(f"\nPreparing transaction matrix (min_support={min_support})...")
        
        # Encode transactions (optimized with sparse=False for better performance)
        te = TransactionEncoder()
        te_array = te.fit(self.transactions).transform(self.transactions, sparse=False)
        
        # Convert to DataFrame with optimized dtype (bool uses less memory)
        self.transaction_matrix = pd.DataFrame(te_array, columns=te.columns_, dtype=bool)
        
        print(f"[OK] Transaction matrix created: {self.transaction_matrix.shape}")
        print(f"  Total unique items: {len(te.columns_)}")
        
        return self.transaction_matrix
    
    def find_frequent_itemsets(self, min_support=0.02, algorithm='apriori', use_colnames=True):
        """
        Find frequent itemsets
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold
        algorithm : str
            Algorithm to use ('apriori' or 'fpgrowth')
        use_colnames : bool
            Whether to use column names in results
        """
        if apriori is None or fpgrowth is None:
            raise ImportError("mlxtend is required. Install: pip install mlxtend")
        
        if self.transaction_matrix is None:
            raise ValueError("No transaction matrix. Please run prepare_transaction_matrix() first.")
        
        print(f"\nFinding frequent itemsets (algorithm={algorithm}, min_support={min_support})...")
        
        try:
            if algorithm == 'apriori':
                self.frequent_itemsets = apriori(
                    self.transaction_matrix, 
                    min_support=min_support, 
                    use_colnames=use_colnames
                )
            elif algorithm == 'fpgrowth':
                self.frequent_itemsets = fpgrowth(
                    self.transaction_matrix,
                    min_support=min_support,
                    use_colnames=use_colnames
                )
            else:
                raise ValueError("Algorithm must be 'apriori' or 'fpgrowth'")
            
            # Immediately ensure support column is numeric (right after getting results)
            # This prevents any downstream type comparison errors
            if 'support' in self.frequent_itemsets.columns:
                # Check current dtype
                if self.frequent_itemsets['support'].dtype == 'object':
                    # If it's object type, there might be mixed types - convert carefully
                    self.frequent_itemsets['support'] = pd.to_numeric(
                        self.frequent_itemsets['support'], 
                        errors='coerce'
                    )
                else:
                    # Even if it's numeric, ensure it's float64 to avoid any issues
                    self.frequent_itemsets['support'] = pd.to_numeric(
                        self.frequent_itemsets['support'], 
                        errors='coerce'
                    ).astype('float64')
            
            # Remove any rows with invalid support values immediately
            initial_count = len(self.frequent_itemsets)
            self.frequent_itemsets = self.frequent_itemsets.dropna(subset=['support'])
            if len(self.frequent_itemsets) < initial_count:
                print(f"  ⚠ Removed {initial_count - len(self.frequent_itemsets)} itemsets with invalid support values")
            
            print(f"[OK] Found {len(self.frequent_itemsets)} frequent itemsets")
            if len(self.frequent_itemsets) > 0:
                print(f"  Max itemset size: {self.frequent_itemsets['itemsets'].apply(len).max()}")
                print(f"  Support range: {self.frequent_itemsets['support'].min():.4f} - {self.frequent_itemsets['support'].max():.4f}")
            
            # Display top itemsets
            print("\nTop 10 Frequent Itemsets:")
            # Filter out rows where support is NaN before sorting
            valid_itemsets = self.frequent_itemsets.dropna(subset=['support'])
            if len(valid_itemsets) > 0:
                top_itemsets = valid_itemsets.nlargest(10, 'support')
                for idx, row in top_itemsets.iterrows():
                    items = ', '.join(list(row['itemsets']))
                    support_value = row['support'] if pd.notna(row['support']) else 0.0
                    print(f"  {items}: support={support_value:.4f}")
            else:
                print("  No valid itemsets with numeric support values")
            
            return self.frequent_itemsets
        except Exception as e:
            print(f"Error finding frequent itemsets: {e}")
            raise
    
    def generate_association_rules(self, metric='confidence', min_threshold=0.5, 
                                   min_support=None):
        """
        Generate association rules from frequent itemsets
        
        Parameters:
        -----------
        metric : str
            Metric to use ('confidence', 'lift', 'support', etc.)
        min_threshold : float
            Minimum threshold for the metric
        min_support : float, optional
            Minimum support threshold
        """
        if association_rules is None:
            raise ImportError("mlxtend is required. Install: pip install mlxtend")
        
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            raise ValueError("No frequent itemsets found. Please run find_frequent_itemsets() first.")
        
        print(f"\nGenerating association rules (metric={metric}, min_threshold={min_threshold})...")
        
        try:
            # Clean and ensure numeric types BEFORE passing to association_rules
            # This prevents type comparison errors inside mlxtend
            frequent_itemsets_clean = self.frequent_itemsets.copy()
            
            # Ensure support column is numeric and remove invalid rows
            if 'support' in frequent_itemsets_clean.columns:
                frequent_itemsets_clean['support'] = pd.to_numeric(
                    frequent_itemsets_clean['support'], 
                    errors='coerce'
                )
                # Remove rows with NaN support values
                frequent_itemsets_clean = frequent_itemsets_clean.dropna(subset=['support'])
            
            # Ensure itemsets column contains proper frozensets (mlxtend requires this)
            # The itemsets should already be frozensets from apriori/fpgrowth, but verify
            if 'itemsets' in frequent_itemsets_clean.columns:
                # Check if itemsets are already frozensets (they should be from mlxtend)
                sample_itemset = frequent_itemsets_clean['itemsets'].iloc[0] if len(frequent_itemsets_clean) > 0 else None
                if sample_itemset is not None and not isinstance(sample_itemset, frozenset):
                    # Only convert if they're not already frozensets
                    def ensure_frozenset(x):
                        if isinstance(x, frozenset):
                            return x
                        elif isinstance(x, (list, tuple, set)):
                            return frozenset(x)
                        else:
                            return frozenset([str(x)])
                    
                    frequent_itemsets_clean['itemsets'] = frequent_itemsets_clean['itemsets'].apply(ensure_frozenset)
            
            # Check if we have valid itemsets
            if len(frequent_itemsets_clean) == 0:
                raise ValueError("No valid frequent itemsets after cleaning. All itemsets had invalid support values.")
            
            # Now generate association rules with cleaned data
            try:
                self.association_rules = association_rules(
                    frequent_itemsets_clean,
                    metric=metric,
                    min_threshold=min_threshold
                )
            except TypeError as e:
                # If we get a type error, it might be due to mixed types in the DataFrame
                # Let's try to diagnose and fix
                print(f"Type error detected: {e}")
                print(f"Frequent itemsets dtypes:\n{frequent_itemsets_clean.dtypes}")
                print(f"Sample support values: {frequent_itemsets_clean['support'].head()}")
                # Try converting all numeric columns explicitly
                for col in frequent_itemsets_clean.select_dtypes(include=[object]).columns:
                    if col != 'itemsets':  # Don't convert itemsets column
                        try:
                            frequent_itemsets_clean[col] = pd.to_numeric(frequent_itemsets_clean[col], errors='coerce')
                        except:
                            pass
                # Retry
                self.association_rules = association_rules(
                    frequent_itemsets_clean,
                    metric=metric,
                    min_threshold=min_threshold
                )
            
            # Immediately ensure all numeric columns are numeric after generation
            if len(self.association_rules) > 0:
                # Convert all numeric columns to proper numeric types
                numeric_cols = ['support', 'confidence', 'lift', 'conviction', 'leverage']
                for col in numeric_cols:
                    if col in self.association_rules.columns:
                        self.association_rules[col] = pd.to_numeric(self.association_rules[col], errors='coerce')
                
                # Remove any rows where the metric column is NaN or non-numeric
                if metric in self.association_rules.columns:
                    self.association_rules[metric] = pd.to_numeric(self.association_rules[metric], errors='coerce')
                    self.association_rules = self.association_rules.dropna(subset=[metric])
            
            if len(self.association_rules) > 0:
                print(f"[OK] Generated {len(self.association_rules)} association rules")
                
                # Ensure numeric columns are numeric type
                numeric_cols = ['support', 'confidence', 'lift', 'conviction', 'leverage']
                for col in numeric_cols:
                    if col in self.association_rules.columns:
                        self.association_rules[col] = pd.to_numeric(self.association_rules[col], errors='coerce')
                
                # Ensure metric column is numeric
                if metric in self.association_rules.columns:
                    self.association_rules[metric] = pd.to_numeric(self.association_rules[metric], errors='coerce')
                
                # Display top rules
                print("\nTop 10 Association Rules:")
                # Filter out rows where metric is NaN before sorting
                valid_rules = self.association_rules.dropna(subset=[metric])
                if len(valid_rules) > 0:
                    top_rules = valid_rules.nlargest(10, metric)
                    for idx, row in top_rules.iterrows():
                        antecedents = ', '.join(list(row['antecedents']))
                        consequents = ', '.join(list(row['consequents']))
                        metric_value = row[metric] if pd.notna(row[metric]) else 0.0
                        support_value = row['support'] if pd.notna(row['support']) else 0.0
                        lift_value = row['lift'] if pd.notna(row['lift']) else 0.0
                        print(f"  {antecedents} -> {consequents}")
                        print(f"    {metric}={metric_value:.4f}, support={support_value:.4f}, lift={lift_value:.4f}")
                else:
                    print("  No valid rules with numeric metric values")
            else:
                print("⚠ No association rules found with given threshold")
            
            return self.association_rules
        except Exception as e:
            print(f"Error generating association rules: {e}")
            raise
    
    def analyze_rules(self):
        """Analyze and extract insights from association rules"""
        if self.association_rules is None or len(self.association_rules) == 0:
            print("No association rules to analyze")
            return {}
        
        print("\nAnalyzing association rules...")
        
        insights = {}
        
        # Ensure numeric columns are numeric type
        numeric_cols = ['support', 'confidence', 'lift', 'conviction', 'leverage']
        for col in numeric_cols:
            if col in self.association_rules.columns:
                self.association_rules[col] = pd.to_numeric(self.association_rules[col], errors='coerce')
        
        # High confidence rules
        if 'confidence' in self.association_rules.columns:
            high_confidence = self.association_rules[
                pd.to_numeric(self.association_rules['confidence'], errors='coerce') >= 0.7
            ]
            insights['high_confidence_rules'] = len(high_confidence)
            print(f"  High confidence rules (≥0.7): {len(high_confidence)}")
        
        # High lift rules
        if 'lift' in self.association_rules.columns:
            high_lift = self.association_rules[
                pd.to_numeric(self.association_rules['lift'], errors='coerce') >= 2.0
            ]
            insights['high_lift_rules'] = len(high_lift)
            print(f"  High lift rules (≥2.0): {len(high_lift)}")
        
        # Most common antecedents
        all_antecedents = []
        for itemset in self.association_rules['antecedents']:
            all_antecedents.extend(list(itemset))
        
        if all_antecedents:
            from collections import Counter
            antecedent_counts = Counter(all_antecedents)
            top_antecedents = antecedent_counts.most_common(5)
            insights['top_antecedents'] = top_antecedents
            print(f"\n  Top 5 Antecedents:")
            for item, count in top_antecedents:
                print(f"    {item}: {count} rules")
        
        # Most common consequents
        all_consequents = []
        for itemset in self.association_rules['consequents']:
            all_consequents.extend(list(itemset))
        
        if all_consequents:
            from collections import Counter
            consequent_counts = Counter(all_consequents)
            top_consequents = consequent_counts.most_common(5)
            insights['top_consequents'] = top_consequents
            print(f"\n  Top 5 Consequents:")
            for item, count in top_consequents:
                print(f"    {item}: {count} rules")
        
        self.insights = insights
        return insights
    
    def get_detailed_summary(self):
        """
        Get a detailed summary of the association rules mining results
        
        Returns:
        --------
        dict: Summary statistics and verification metrics
        """
        summary = {
            'algorithm_status': 'success',
            'transactions_loaded': len(self.transactions) if self.transactions else 0,
            'unique_items': self.transaction_matrix.shape[1] if self.transaction_matrix is not None else 0,
            'frequent_itemsets_count': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            'association_rules_count': len(self.association_rules) if self.association_rules is not None else 0,
        }
        
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            summary['frequent_itemsets_stats'] = {
                'max_itemset_size': int(self.frequent_itemsets['itemsets'].apply(len).max()),
                'min_support': float(self.frequent_itemsets['support'].min()),
                'max_support': float(self.frequent_itemsets['support'].max()),
                'avg_support': float(self.frequent_itemsets['support'].mean()),
            }
        
        if self.association_rules is not None and len(self.association_rules) > 0:
            summary['association_rules_stats'] = {}
            numeric_cols = ['support', 'confidence', 'lift', 'conviction', 'leverage']
            for col in numeric_cols:
                if col in self.association_rules.columns:
                    summary['association_rules_stats'][f'{col}_min'] = float(self.association_rules[col].min())
                    summary['association_rules_stats'][f'{col}_max'] = float(self.association_rules[col].max())
                    summary['association_rules_stats'][f'{col}_mean'] = float(self.association_rules[col].mean())
        
        return summary
    
    def save_results(self, output_dir='association_rules_output'):
        """Save all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving results to {output_path}...")
        
        if self.frequent_itemsets is not None:
            # Convert itemsets to string for CSV
            itemsets_df = self.frequent_itemsets.copy()
            itemsets_df['itemsets'] = itemsets_df['itemsets'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            # Ensure all columns are saved
            itemsets_df.to_csv(output_path / 'frequent_itemsets.csv', index=False)
            print(f"  [OK] Saved frequent_itemsets.csv ({len(itemsets_df)} itemsets)")
        
        if self.association_rules is not None and len(self.association_rules) > 0:
            # Convert itemsets to string for CSV - preserve ALL columns
            rules_df = self.association_rules.copy()
            rules_df['antecedents'] = rules_df['antecedents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_df['consequents'] = rules_df['consequents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            # Save all columns including all metrics
            rules_df.to_csv(output_path / 'association_rules.csv', index=False)
            print(f"  [OK] Saved association_rules.csv ({len(rules_df)} rules, {len(rules_df.columns)} columns)")
        
        if self.insights:
            import json
            with open(output_path / 'insights.json', 'w') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print(f"  [OK] Saved insights.json")
        
        print(f"[OK] All results saved to {output_path}")
    
    def run_complete_analysis(self, min_support=0.02, min_confidence=0.5, 
                             sample_size=None, output_dir='association_rules_output'):
        """
        Run complete association rules mining pipeline
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold
        min_confidence : float
            Minimum confidence threshold
        sample_size : int, optional
            Number of transactions to sample
        output_dir : str
            Directory to save results
        """
        print("\n" + "=" * 80)
        print("COMPLETE ASSOCIATION RULES MINING PIPELINE")
        print("=" * 80)
        
        # Load transactions
        self.load_transactions(sample_size=sample_size)
        
        # Prepare transaction matrix
        self.prepare_transaction_matrix(min_support=min_support)
        
        # Find frequent itemsets
        self.find_frequent_itemsets(min_support=min_support, algorithm='apriori')
        
        # Generate association rules
        self.generate_association_rules(metric='confidence', min_threshold=min_confidence)
        
        # Analyze rules
        self.analyze_rules()
        
        # Save results
        self.save_results(output_dir)
        
        print("\n" + "=" * 80)
        print("ASSOCIATION RULES MINING COMPLETE!")
        print("=" * 80)
        
        return {
            'frequent_itemsets': self.frequent_itemsets,
            'association_rules': self.association_rules,
            'insights': self.insights
        }


# Example usage
if __name__ == "__main__":
    # Initialize association rules mining
    arm = AssociationRulesMining()
    
    # Run complete analysis (using sample for faster execution)
    results = arm.run_complete_analysis(
        min_support=0.02,
        min_confidence=0.5,
        sample_size=50000  # Use sample for testing
    )
    
    print("\nAssociation Rules Mining completed successfully!")

