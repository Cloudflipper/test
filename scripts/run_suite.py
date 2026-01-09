"""Script to run all models in the suite and aggregate results."""
import argparse
import os
import sys
import csv
import json
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.io import load_config
from utils.logger import setup_logger


def run_single_model(model_name: str, base_config: str, dataset_name: str, logger):
    """Run a single model and return metrics."""
    logger.info(f"Running {model_name}...")
    
    cmd = [
        sys.executable,
        "scripts/run_single.py",
        "--config", base_config,
        "--model", model_name,
        "--dataset", dataset_name
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        if result.returncode != 0:
            if result.returncode == 2:
                # Model not integrated - this is expected
                logger.info(f"Model '{model_name}' is not integrated yet.")
            else:
                logger.error(f"Error running {model_name}:")
                logger.error(result.stderr)
            return None
        
        # Parse metrics from output (simplified - in practice you might want to return JSON)
        # For now, we'll extract from the log or use a results file
        # This is a simplified version - you might want to modify run_single.py to save results
        logger.info(f"Completed {model_name}")
        return {"status": "completed"}
    
    except Exception as e:
        logger.error(f"Exception running {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run suite of models and aggregate results")
    parser.add_argument("--suite", type=str, default="configs/suite_10.yaml",
                        help="Path to suite configuration file")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base configuration file")
    args = parser.parse_args()
    
    # Load suite configuration
    suite_config = load_config(args.suite)
    base_config = args.config
    
    # Setup logger
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("run_suite", log_dir)
    
    # Check for empty models list
    models_list = suite_config.get("models", [])
    if not models_list:
        print("No models specified in suite configuration.")
        logger.info("No models specified in suite configuration.")
        sys.exit(0)
    
    logger.info("=" * 50)
    logger.info("Starting model suite evaluation")
    logger.info("=" * 50)
    
    # Results storage
    results = []
    
    # Run each model
    for model_entry in models_list:
        # Support both dict format {"name": "...", "config": "..."} and simple string format
        if isinstance(model_entry, dict):
            model_name = model_entry.get("name", model_entry.get("model", ""))
        else:
            model_name = str(model_entry)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*50}\n")
        
        # Run model
        dataset_name = suite_config.get("dataset", "synthetic")
        result = run_single_model(model_name, base_config, dataset_name, logger)
        
        if result:
            results.append({
                "model": model_name,
                "status": result.get("status", "unknown")
            })
        else:
            results.append({
                "model": model_name,
                "status": "failed"
            })
    
    # Aggregate results from individual model runs
    # In a more sophisticated version, you would read results from files saved by run_single.py
    # For now, we'll create a simple CSV with model names and status
    
    # Create results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Write aggregated results
    output_csv = suite_config.get("output_csv", "./results/suite_results.csv")
    
    # Collect actual metrics from individual result files
    csv_data = []
    csv_data.append(["Model", "Status", "MAE", "RMSE", "MAPE"])
    
    for result in results:
        model_name = result["model"]
        status = result["status"]
        
        # Try to read metrics from JSON results file
        results_json = os.path.join(results_dir, f"{model_name}_results.json")
        if os.path.exists(results_json):
            try:
                with open(results_json, 'r') as f:
                    metrics = json.load(f)
                mae = metrics.get("MAE", "N/A")
                rmse = metrics.get("RMSE", "N/A")
                mape = metrics.get("MAPE", "N/A")
                csv_data.append([model_name, status, mae, rmse, mape])
            except Exception as e:
                logger.warning(f"Could not read results for {model_name}: {e}")
                csv_data.append([model_name, status, "N/A", "N/A", "N/A"])
        else:
            csv_data.append([model_name, status, "N/A", "N/A", "N/A"])
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Suite evaluation completed")
    logger.info(f"Results saved to: {output_csv}")
    logger.info(f"{'='*50}\n")
    
    # Print summary
    logger.info("Summary:")
    for result in results:
        logger.info(f"  {result['model']}: {result['status']}")


if __name__ == "__main__":
    main()

