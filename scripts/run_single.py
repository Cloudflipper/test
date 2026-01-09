"""Script to train and evaluate a single model."""
import argparse
import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.io import load_config, merge_configs
from datasets.loader import create_dataloaders
from graphs.adjacency import build_graph
from models.registry import create_model
from engine.trainer import Trainer
from engine.evaluator import Evaluator


def _safe_minmax(t: torch.Tensor):
    t = t.detach()
    return float(t.min().item()), float(t.max().item())


def _log_transform_stats(logger, transform):
    """Try hard to print useful scaler/transform stats (mean/std or min/max)."""
    logger.info("===== Transform Stats (B) =====")
    if transform is None:
        logger.info("transform: None")
        logger.info("================================")
        return

    logger.info(f"transform type: {type(transform)}")

    # Common patterns:
    # 1) sklearn StandardScaler: mean_, scale_
    # 2) sklearn MinMaxScaler: data_min_, data_max_
    # 3) custom object: mean/std/min/max fields
    # 4) dict-like payloads
    # 5) wrapper that holds .scaler

    candidates = [("transform", transform)]
    if hasattr(transform, "scaler"):
        candidates.append(("transform.scaler", getattr(transform, "scaler")))

    # If transform is dict-like
    if isinstance(transform, dict):
        for k, v in transform.items():
            candidates.append((f"transform['{k}']", v))

    printed_any = False

    def _print_field(obj_name, obj, field_name):
        nonlocal printed_any
        if hasattr(obj, field_name):
            val = getattr(obj, field_name)
            # numpy / torch / list
            try:
                if torch.is_tensor(val):
                    vmin, vmax = _safe_minmax(val)
                    logger.info(f"{obj_name}.{field_name}: tensor range {vmin:.6f} ~ {vmax:.6f}")
                else:
                    # try numpy-ish
                    import numpy as np
                    arr = np.array(val)
                    if arr.size > 0 and arr.dtype != object:
                        logger.info(f"{obj_name}.{field_name}: shape={arr.shape}, "
                                    f"range={float(arr.min()):.6f} ~ {float(arr.max()):.6f}")
                    else:
                        logger.info(f"{obj_name}.{field_name}: {val}")
                printed_any = True
            except Exception:
                logger.info(f"{obj_name}.{field_name}: {val}")
                printed_any = True

    for name, obj in candidates:
        # StandardScaler-like
        _print_field(name, obj, "mean_")
        _print_field(name, obj, "scale_")
        _print_field(name, obj, "var_")

        # MinMaxScaler-like
        _print_field(name, obj, "data_min_")
        _print_field(name, obj, "data_max_")
        _print_field(name, obj, "min_")
        _print_field(name, obj, "max_")

        # Custom common names
        _print_field(name, obj, "mean")
        _print_field(name, obj, "std")
        _print_field(name, obj, "min")
        _print_field(name, obj, "max")

    if not printed_any:
        # As a last resort, show available attribute names (trimmed)
        attrs = [a for a in dir(transform) if not a.startswith("_")]
        logger.info("No known scaler fields found. Public attrs (first 50):")
        logger.info(", ".join(attrs[:50]))

    logger.info("================================")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a single model")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base configuration file")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="Dataset name")
    args = parser.parse_args()

    # Load base configuration
    base_config = load_config(args.config)
    model_name = args.model.lower()
    dataset_name = args.dataset

    # Load model-specific config if it exists
    model_config_path = f"configs/models/{model_name}.yaml"
    if os.path.exists(model_config_path):
        model_config = load_config(model_config_path)
        # Deep merge model config into base config
        config = merge_configs(base_config, model_config)
    else:
        config = base_config

    # Setup
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Create directories
    checkpoint_dir = os.path.join(config["paths"]["checkpoint_dir"], model_name)
    log_dir = config["paths"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(f"run_single_{model_name}", log_dir)
    logger.info(f"Running model: {model_name}")
    logger.info(f"Device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataset_name = args.dataset

    # Prepare dataloader arguments
    dataloader_kwargs = {
        "dataset_name": dataset_name,
        "seq_len": config["dataset"]["seq_len"],
        "pred_len": config["dataset"]["pred_len"],
        "train_ratio": config["dataset"]["train_ratio"],
        "val_ratio": config["dataset"]["val_ratio"],
        "test_ratio": config["dataset"]["test_ratio"],
        "batch_size": config["training"]["batch_size"],
        "seed": config["seed"],
        "data_dir": config["dataset"].get("root", "./data")
    }

    # Add dataset-specific arguments
    if dataset_name == "synthetic":
        dataloader_kwargs["num_samples"] = 1000  # Can be made configurable
        dataloader_kwargs["num_nodes"] = config["dataset"]["num_nodes"]
        dataloader_kwargs["num_features"] = config["dataset"]["num_features"]
    elif dataset_name == "los":
        # LOS dataset parameters are inferred from data files
        pass

    train_loader, val_loader, test_loader, transform = create_dataloaders(**dataloader_kwargs)

    # ===== A/B sanity checks (your request) =====
    # A: y(norm) range
    logger.info("===== Quick Sanity Check (A) =====")
    try:
        _x0, _y0 = next(iter(test_loader))
        # y should be [B, Tout, N, F] or [B, Tout, N] depending on dataset wrapper
        if torch.is_tensor(_y0):
            y_min, y_max = _safe_minmax(_y0)
            logger.info(f"A: y(norm) range: {y_min:.6f} ~ {y_max:.6f} | y shape: {tuple(_y0.shape)}")
        else:
            logger.info(f"A: y is not a torch.Tensor. type={type(_y0)}")
    except Exception as e:
        logger.info(f"A: failed to read a test batch for y(norm) range: {repr(e)}")
    logger.info("==================================")

    # B: transform stats
    _log_transform_stats(logger, transform)
    # ===========================================

    # Build graph
    logger.info("Building graph...")
    graph_type = config["graph"]["type"]

    # Prepare graph arguments
    graph_kwargs = {
        "graph_type": graph_type,
        "seed": config["seed"]
    }

    if graph_type == "synthetic":
        graph_kwargs["num_nodes"] = config["dataset"]["num_nodes"]
        graph_kwargs["self_loop"] = config["graph"]["self_loop"]
        graph_kwargs["normalize"] = config["graph"]["normalize"]
        graph_kwargs["return_raw"] = False
    elif graph_type == "los":
        # LOS graph is loaded from CSV, already normalized
        graph_kwargs["adj_path"] = os.path.join(config["dataset"].get("root", "./data"), "los_adj.csv")

    adj = build_graph(**graph_kwargs)
    logger.info(f"Adj shape: {tuple(adj.shape)}")
    logger.info(f"Adj min/max: {adj.min().item():.6f} / {adj.max().item():.6f}")

    # Create model
    logger.info("Creating model...")
    try:
        # Infer dataset dimensions from first batch if needed
        if dataset_name == "los":
            # Get dimensions from first batch
            sample_x, _ = next(iter(train_loader))
            _, _, num_nodes, num_features = sample_x.shape
            logger.info(f"LOS dataset: {num_nodes} nodes, {num_features} features")
        else:
            num_nodes = config["dataset"]["num_nodes"]
            num_features = config["dataset"]["num_features"]

        # Prepare model config with dataset parameters
        model_cfg = {
            "num_nodes": num_nodes,
            "num_features": num_features,
            "seq_len": config["dataset"]["seq_len"],
            "pred_len": config["dataset"]["pred_len"],
        }
        # Add model-specific hyperparameters from config["model"]
        if "model" in config and isinstance(config["model"], dict):
            model_cfg.update(config["model"])

        model = create_model(model_name, model_cfg)
        model = model.to(device)


       
    except NotImplementedError as e:
        print(f"Model '{model_name}' is not integrated yet. Provide source code to integrate.")
        sys.exit(2)
    
    model.train()
    with torch.no_grad():
        sample_x, _ = next(iter(train_loader))
        sample_x = sample_x.to(device)
        _ = model(sample_x, adj=adj.to(device))     
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adj=adj,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        early_stopping_patience=config["training"]["early_stopping"]["patience"],
        early_stopping_min_delta=config["training"]["early_stopping"]["min_delta"],
        gradient_clip=config["training"]["gradient_clip"],
        logger=logger
    )

    # Train
    logger.info("Starting training...")
    trainer.train(epochs=config["training"]["epochs"])

    # Load best model
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_checkpoint):
        trainer.load_checkpoint(best_checkpoint)
        logger.info("Loaded best model for evaluation")

    # Evaluate
    logger.info("Evaluating on test set...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        adj=adj,
        device=device,
        transform=transform,
        logger=logger
    )

    test_metrics = evaluator.evaluate()

    logger.info("=" * 50)
    logger.info("Final Test Results:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")
    logger.info("=" * 50)

    # Save results to file
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")

    # Also save as JSON for easier parsing
    import json
    results_json = os.path.join(results_dir, f"{model_name}_results.json")
    with open(results_json, 'w') as f:
        json.dump({"model": model_name, **test_metrics}, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    main()
