"""
Standalone script to run HRP analysis with custom parameters.
Usage: python run_analysis.py [--config config.yaml]
"""

import argparse
import yaml
from pathlib import Path
import logging
from hrp import main, Config, config

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config_from_yaml(yaml_config: dict):
    """Update global config from YAML."""
    if 'data' in yaml_config:
        config.start_date = yaml_config['data'].get('start_date', config.start_date)
        config.end_date = yaml_config['data'].get('end_date', config.end_date)
        config.min_data_pct = yaml_config['data'].get('min_data_pct', config.min_data_pct)
    
    if 'walk_forward' in yaml_config:
        config.train_window = yaml_config['walk_forward'].get('train_window', config.train_window)
        config.test_window = yaml_config['walk_forward'].get('test_window', config.test_window)
    
    if 'transaction_costs' in yaml_config:
        config.cost_bps = yaml_config['transaction_costs'].get('cost_bps', config.cost_bps)
        config.rebalance_freq = yaml_config['transaction_costs'].get('rebalance_freq', config.rebalance_freq)
    
    if 'statistical_tests' in yaml_config:
        config.bootstrap_samples = yaml_config['statistical_tests'].get('bootstrap_samples', config.bootstrap_samples)
        config.confidence_level = yaml_config['statistical_tests'].get('confidence_level', config.confidence_level)
    
    if 'output' in yaml_config:
        config.output_dir = Path(yaml_config['output'].get('directory', config.output_dir))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Hierarchical Risk Parity portfolio optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default='sample_data/financials.csv',
        help='Path to CSV file with tickers'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides config'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), overrides config'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory, overrides config'
    )
    
    parser.add_argument(
        '--no-walk-forward',
        action='store_true',
        help='Skip walk-forward analysis (faster)'
    )
    
    parser.add_argument(
        '--no-costs',
        action='store_true',
        help='Skip transaction cost analysis'
    )
    
    parser.add_argument(
        '--no-tests',
        action='store_true',
        help='Skip statistical tests'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip walk-forward, costs, and tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main_cli():
    """Main CLI entry point."""
    args = parse_args()
    
    # Load config if exists
    if Path(args.config).exists():
        yaml_config = load_config(args.config)
        update_config_from_yaml(yaml_config)
        print(f"✓ Loaded configuration from {args.config}")
    else:
        print(f"⚠ Config file {args.config} not found, using defaults")
        yaml_config = {}
    
    # Command line arguments override config
    csv_path = args.csv
    start_date = args.start_date or config.start_date
    end_date = args.end_date or config.end_date
    output_dir = args.output_dir or str(config.output_dir)
    
    # Quick mode settings
    if args.quick:
        args.no_walk_forward = True
        args.no_costs = True
        args.no_tests = True
        print("⚡ Quick mode enabled: skipping walk-forward, costs, and tests")
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("HRP PORTFOLIO OPTIMIZATION")
    print("="*70)
    print(f"Data source: {csv_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")
    print(f"Walk-forward: {'No' if args.no_walk_forward else 'Yes'}")
    print(f"Transaction costs: {'No' if args.no_costs else 'Yes'}")
    print(f"Statistical tests: {'No' if args.no_tests else 'Yes'}")
    print("="*70 + "\n")
    
    # Import and modify main function behavior
    import hrp as hrp_module
    
    # Override run_complete_comparison parameters
    original_func = hrp_module.run_complete_comparison
    
    def modified_comparison(returns, **kwargs):
        # Remove duplicates in case callers already supply these flags
        kwargs.pop('include_costs', None)
        kwargs.pop('walk_forward', None)
        kwargs.pop('statistical_tests', None)
        
        return original_func(
            returns,
            include_costs=not args.no_costs,
            walk_forward=not args.no_walk_forward,
            statistical_tests=not args.no_tests,
            **kwargs
        )
    
    hrp_module.run_complete_comparison = modified_comparison
    
    # Run main analysis
    try:
        weights, results = main(
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir
        )
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}/")
        print("\nTop 10 Holdings:")
        print(weights.sort_values(ascending=False).head(10).to_string())
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main_cli())
