"""Evaluate a single model checkpoint on the test set (no training)."""
import argparse
import os
import sys
import json
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.io import load_config, merge_configs
from datasets.loader import create_dataloaders
from graphs.adjacency import build_graph
from models.registry import create_model
from engine.evaluator import Evaluator


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load a checkpoint saved by Trainer into the model."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both formats:
    # 1) {"model_state_dict": ...}
    # 2) direct state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model on test set")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base configuration file")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., tgcn)")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="Dataset name (synthetic/los/sz etc.)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth (default: best_model.pth under checkpoints/<model>/)")
    args = parser.parse_args()

    model_name = args.model.lower()
    dataset_name_cli = args.dataset

    # Load base config + optional model config
    base_config = load_config(args.config)
    model_config_path = f"configs/models/{model_name}.yaml"
    if os.path.exists(model_config_path):
        model_config = load_config(model_config_path)
        config = merge_configs(base_config, model_config)
    else:
        config = base_config

    # Setup seed + device
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Directories
    log_dir = config["paths"]["log_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Logger
    logger = setup_logger(f"eval_single_{model_name}", log_dir)
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"Device: {device}")

    # Determine dataset name from config if present
    dataset_name = dataset_name_cli



    # Create dataloaders (must return transform for inverse scaling)
    logger.info("Creating dataloaders...")
    dataloader_kwargs = {
        "dataset_name": dataset_name,
        "seq_len": config["dataset"]["seq_len"],
        "pred_len": config["dataset"]["pred_len"],
        "train_ratio": config["dataset"]["train_ratio"],
        "val_ratio": config["dataset"]["val_ratio"],
        "test_ratio": config["dataset"]["test_ratio"],
        "batch_size": config["training"]["batch_size"],
        "seed": config["seed"],
        "data_dir": config["dataset"].get("root", "./data"),
    }

    if dataset_name == "synthetic":
        dataloader_kwargs["num_samples"] = 1000
        dataloader_kwargs["num_nodes"] = config["dataset"]["num_nodes"]
        dataloader_kwargs["num_features"] = config["dataset"]["num_features"]
    elif dataset_name == "los":
        # inferred from CSVs in loader; no extra args needed
        pass

    train_loader, val_loader, test_loader, transform = create_dataloaders(**dataloader_kwargs)

    # Build graph
    logger.info("Building graph...")
    graph_type = config["graph"]["type"]
    graph_kwargs = {"graph_type": graph_type, "seed": config["seed"]}

    if graph_type == "synthetic":
        graph_kwargs["num_nodes"] = config["dataset"]["num_nodes"]
        graph_kwargs["self_loop"] = config["graph"]["self_loop"]
        graph_kwargs["normalize"] = config["graph"]["normalize"]
        graph_kwargs["return_raw"] = False
    elif graph_type == "los":
        graph_kwargs["adj_path"] = os.path.join(config["dataset"].get("root", "./data"), "los_adj.csv")

    adj = build_graph(**graph_kwargs)

    # Infer num_nodes/num_features from a batch if needed
    sample_x, _ = next(iter(test_loader))
    _, _, num_nodes, num_features = sample_x.shape
    logger.info(f"{dataset_name.upper()} dataset: {num_nodes} nodes, {num_features} features")

    # Create model
    logger.info("Creating model...")
    model_cfg = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "seq_len": config["dataset"]["seq_len"],
        "pred_len": config["dataset"]["pred_len"],
    }
    if "model" in config and isinstance(config["model"], dict):
        model_cfg.update(config["model"])

    model = create_model(model_name, model_cfg).to(device)


    
    # Load checkpoint
    if args.checkpoint is None:
        ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], model_name, "best_model.pth")
    else:
        ckpt_path = args.checkpoint

    logger.info(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, device=device)
    logger.info("Checkpoint loaded.")

    # Evaluate
    logger.info("Evaluating on test set...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        adj=adj,
        device=device,
        transform=transform,
        logger=logger,
    )
    test_metrics = evaluator.evaluate()

    # Print + save
    logger.info("=" * 50)
    logger.info("Final Eval Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("=" * 50)

    txt_path = os.path.join(results_dir, f"{model_name}_eval_results.txt")
    json_path = os.path.join(results_dir, f"{model_name}_eval_results.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, "dataset": dataset_name, "checkpoint": ckpt_path, **test_metrics},
            f,
            indent=2,
        )

    logger.info(f"Saved results to: {txt_path}")
    logger.info(f"Saved results to: {json_path}")


if __name__ == "__main__":
    main()
