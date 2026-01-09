# Models Integration Guide

## Directory Structure

Models are organized in subdirectories under `models/`:

```
models/
├── registry.py          # Model factory/registry
├── base.py              # Base interface (optional)
├── README.md            # This file
└── <model_name>/        # Model-specific directory
    ├── __init__.py      # Exports the wrapper/model class
    ├── original.py      # Original model implementation (unmodified)
    ├── wrapper.py       # Wrapper adapting to benchmark interface
    └── [other files]    # Model-specific utilities, helpers, etc.
```

### Example: TGCN Structure
```
models/
└── tgcn/
    ├── __init__.py      # Exports TGCNWrapper
    ├── original.py      # Original TGCN implementation
    ├── wrapper.py       # TGCNWrapper adapting to benchmark
    └── graph_conv.py    # TGCN-specific graph utilities
```

## Integration Contract

All models must implement:

### Input / Output
- Input `x`: float tensor shaped **[B, Tin, N, F]**
  - B: batch size
  - Tin: input sequence length
  - N: number of nodes
  - F: number of features
- Optional `adj`: adjacency matrix (dense [N,N] or sparse), if the model uses a graph
- Output `y_hat`: float tensor shaped **[B, Tout, N, 1]**
  - B: batch size
  - Tout: prediction sequence length
  - N: number of nodes
  - 1: single feature (target variable)

### Interface
Models should implement the `forward` method:
```python
def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
    """Forward pass."""
    # x: [B, Tin, N, F]
    # adj: [N, N] (optional)
    # Returns: [B, Tout, N, 1]
    pass
```

## Integration Steps

1. **Create model subdirectory**: `models/<model_name>/`

2. **Add original implementation**: Copy original model code to `original.py` (do not modify logic)

3. **Create wrapper**: Implement `wrapper.py` that:
   - Adapts input/output shapes to benchmark interface
   - Handles multi-feature input if needed (project F → 1)
   - Implements multi-step forecasting if model only does single-step
   - Wraps original model without modifying its logic

4. **Create __init__.py**: Export the wrapper/model class

5. **Register in registry.py**: Add import and conditional in `create_model()` function

6. **Add config file**: Create `configs/models/<model_name>.yaml` with model hyperparameters

## Notes
- Scaling is handled outside the model (trainer/evaluator).
- Horizon Tout is configured via YAML config.
- Original model implementations should remain unmodified (use wrapper pattern).
- Each model is self-contained in its subdirectory.
