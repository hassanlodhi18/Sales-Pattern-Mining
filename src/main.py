"""
Main Entry Point for Sales Data Warehouse Analytics
Alternative to Streamlit GUI - Command Line Interface
"""

import argparse
from pathlib import Path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sales Data Warehouse Analytics Platform"
    )
    
    parser.add_argument(
        'module',
        choices=['kdd', 'forecast', 'association', 'visualize', 'powerbi', 'eda', 'all'],
        help='Module to run'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for processing (optional)'
    )
    
    parser.add_argument(
        '--forecast-steps',
        type=int,
        default=30,
        help='Number of forecast steps (default: 30)'
    )
    
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.02,
        help='Minimum support for association rules (default: 0.02)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence for association rules (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SALES DATA WAREHOUSE ANALYTICS PLATFORM")
    print("=" * 80)
    print()
    
    if args.module == 'kdd' or args.module == 'all':
        print("\n" + "=" * 80)
        print("RUNNING KDD PROCESS")
        print("=" * 80)
        from kdd_process import KDDProcess
        kdd = KDDProcess()
        kdd.run_complete_process(
            sample_size=args.sample_size,
            mining_tasks=['summary', 'patterns', 'correlation']
        )
    
    if args.module == 'forecast' or args.module == 'all':
        print("\n" + "=" * 80)
        print("RUNNING TIME SERIES FORECASTING")
        print("=" * 80)
        from time_series_forecasting import TimeSeriesForecasting
        forecaster = TimeSeriesForecasting()
        forecaster.run_complete_forecasting(forecast_steps=args.forecast_steps)
    
    if args.module == 'association' or args.module == 'all':
        print("\n" + "=" * 80)
        print("RUNNING ASSOCIATION RULES MINING")
        print("=" * 80)
        from association_rules_mining import AssociationRulesMining
        arm = AssociationRulesMining()
        arm.run_complete_analysis(
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            sample_size=args.sample_size
        )
    
    if args.module == 'visualize' or args.module == 'all':
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        from visualizations import AdvancedVisualizations
        viz = AdvancedVisualizations()
        viz.create_all_visualizations()
    
    if args.module == 'powerbi' or args.module == 'all':
        print("\n" + "=" * 80)
        print("PREPARING POWER BI DATA")
        print("=" * 80)
        from powerbi_integration import PowerBIIntegration
        pbi = PowerBIIntegration()
        pbi.prepare_all_for_powerbi()
    
    if args.module == 'eda' or args.module == 'all':
        print("\n" + "=" * 80)
        print("RUNNING EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        from eda_warehouse import main as eda_main
        eda_main()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()

