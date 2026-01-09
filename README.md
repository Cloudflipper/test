# Spatiotemporal Forecasting Benchmark

A unified benchmark codebase for comparing spatiotemporal forecasting models with identical preprocessing, graph construction, training budget, and evaluation metrics.

## Features

- **Clean Skeleton**: Ready-to-integrate benchmark framework with no concrete model implementations
- **Unified Pipeline**: Same preprocessing, graph construction, training, and evaluation for all models
- **Synthetic Dataset**: End-to-end runnable synthetic dataset generator
- **Comprehensive Metrics**: MAE, RMSE, MAPE (with epsilon handling)
- **Graceful Failure**: Clear error messages when models are not yet integrated

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── configs/
│   ├── base.yaml              # Base configuration
│   ├── suite_10.yaml          # Suite configuration for all 10 models
│   └── models/                # Individual model configs
├── datasets/
│   ├── synthetic.py           # Synthetic dataset generator
│   ├── loader.py              # Data loading utilities
│   └── transforms.py          # Data transforms and scaling
├── graphs/
│   └── adjacency.py           # Graph construction and normalization
├── models/
│   ├── registry.py            # Model registry (factory)
│   └── README.md              # Model integration contract
├── engine/
│   ├── trainer.py             # Training engine
│   └── evaluator.py           # Evaluation engine
├── scripts/
│   ├── run_single.py          # Train and evaluate single model
│   └── run_suite.py           # Run all models and aggregate results
└── utils/
    ├── seed.py                # Seed utilities
    ├── logger.py              # Logging utilities
    ├── metrics.py             # Evaluation metrics
    ├── scaler.py              # Data scaling
    └── io.py                  # Configuration I/O
```

## Usage

### Train and Evaluate a Single Model

```powershell
python scripts\run_single.py --config configs\base.yaml --model <model_name> --dataset synthetic
```

**Note**: Models must be integrated first. See `models/README.md` for the integration contract.

### Run All Models in Suite

```powershell
python scripts\run_suite.py --suite configs\suite_10.yaml --config configs\base.yaml
```

Results will be aggregated in `results/suite_results.csv`.

### Adding Models

1. Implement your model following the contract in `models/README.md`
2. Register it in `models/registry.py`
3. Add configuration to `configs/models/` if needed
4. Add to `configs/suite_10.yaml` models list

## Data Format

- **Input**: `x [B, Tin, N, F]` - Batch, Input sequence length, Nodes, Features
- **Output**: `y [B, Tout, N, 1]` - Batch, Prediction length, Nodes, Single feature

## Configuration

Edit `configs/base.yaml` to adjust:
- Dataset parameters (num_nodes, num_features, seq_len, pred_len)
- Graph construction (type, self_loop, normalization)
- Training hyperparameters (batch_size, epochs, learning_rate, etc.)
- Evaluation settings

Edit individual model configs in `configs/models/` to adjust model-specific parameters.

## Metrics

All metrics are computed on inverse-scaled predictions:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error (with epsilon handling)

## Notes

- **Model Integration**: This is a clean skeleton. No models are implemented yet. See `models/README.md` for integration guidelines.
- Scaler is fitted on training data only
- Graph adjacency supports self-loop and symmetric/asymmetric normalization
- Early stopping with configurable patience
- Gradient clipping for training stability
- All models must follow the same interface: `forward(x, adj=None, **kwargs) -> y_hat`
- Models that are not integrated will raise `NotImplementedError` with a clear message

## Verification

See `VERIFICATION.md` for commands to verify the skeleton is working correctly.

