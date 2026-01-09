# TGCN Integration Summary

## Directory Structure

The model is organized in a modular structure for easy extension:

```
models/
├── registry.py          # Model factory/registry
├── base.py              # Base interface (optional)
├── README.md            # Integration guide
└── tgcn/                # TGCN model subdirectory
    ├── __init__.py      # Exports TGCNWrapper
    ├── original.py      # Original TGCN implementation (unmodified)
    ├── wrapper.py        # Wrapper adapting to benchmark interface
    └── graph_conv.py    # TGCN-specific graph utilities
```

## Files Created

### 1. `models/tgcn/graph_conv.py`
- **Source**: Copied from `c:\Users\97239\Downloads\graph_conv.py`
- **Purpose**: Contains `calculate_laplacian_with_self_loop` function used by TGCN
- **Location**: Model-specific subdirectory (keeps TGCN self-contained)

### 2. `models/tgcn/original.py`
- **Source**: Copied from `c:\Users\97239\Downloads\tgcn.py`
- **Purpose**: Original TGCN implementation (unmodified logic)
- **Changes**: Updated import to use `models.tgcn.graph_conv`

### 3. `models/tgcn/wrapper.py`
- **Purpose**: Wrapper that adapts TGCN to benchmark interface
- **Key Features**:
  - **Feature Projection**: Projects multi-feature input (F > 1) to single feature using `Linear(F → 1)`
  - **Graph Handling**: Receives `adj` from benchmark and calls `calculate_laplacian_with_self_loop(adj)` inside wrapper
  - **Lazy Initialization**: Initializes TGCN on first forward pass when `adj` is available
  - **Output Head**: Adds `Linear(hidden_dim → 1)` to produce single-step predictions
  - **Autoregressive Rollout**: Implements multi-step forecasting by:
    1. Processing input sequence through TGCN to get final hidden state
    2. Using output head to get first prediction
    3. Using predictions as inputs for subsequent steps via TGCNCell
    4. Stacking all predictions to produce `[B, Tout, N, 1]` output

### 4. `models/tgcn/__init__.py`
- **Purpose**: Exports TGCNWrapper for clean imports
- **Usage**: Allows `from models.tgcn import TGCNWrapper`

### 5. `models/base.py`
- **Purpose**: Optional base interface for models
- **Note**: Provides abstract base class for type checking and documentation

### 6. `configs/models/tgcn.yaml`
- **Purpose**: Model-specific configuration
- **Contents**: Only model hyperparameters (`hidden_dim: 64`)
- **Note**: Does not include dataset or training parameters (handled by base config)

## Files Modified

### 1. `models/registry.py`
- **Changes**:
  - Added import: `from models.tgcn import TGCNWrapper`
  - Added conditional check: `if name == "tgcn": return TGCNWrapper(**cfg)`
  - Kept `NotImplementedError` for unknown models
  - Updated comments to reflect modular structure

### 2. `models/README.md`
- **Changes**: Updated with new directory structure and integration guide

## Integration Details

### Input/Output Contract
- **Input**: `x [B, Tin, N, F]`, `adj [N, N]`
- **Output**: `y_hat [B, Tout, N, 1]`

### Key Adaptations
1. **Multi-feature handling**: If `F > 1`, projects to single feature before feeding to TGCN
2. **Graph normalization**: Laplacian computation happens inside wrapper (not pre-normalized)
3. **Multi-step forecasting**: Autoregressive rollout using TGCNCell for `Tout` steps
4. **Device handling**: Properly moves model to same device as inputs

### Usage
```bash
python scripts\run_single.py --config configs\base.yaml --model tgcn --dataset synthetic
```

## Benefits of New Structure

1. **Modular Organization**: Each model is self-contained in its own subdirectory
2. **Easy Extension**: Adding new models follows the same pattern:
   - Create `models/<new_model>/` directory
   - Add original implementation, wrapper, and utilities
   - Register in `registry.py`
3. **Clear Separation**: Model-specific code is isolated from shared infrastructure
4. **Maintainability**: Easy to locate and modify model-specific code

## Verification

The integration maintains:
- ✅ Original TGCN logic unchanged (via wrapper pattern)
- ✅ Benchmark interface compliance
- ✅ Graceful error handling for unknown models
- ✅ No modifications to datasets/, engine/, graphs/, utils/ infrastructure
- ✅ Modular structure ready for additional models

