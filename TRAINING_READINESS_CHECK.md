# TGCN Training Readiness Check

## 1. Training Prerequisites

### a) Environment (Python, torch, numpy, etc.)

**Required Components:**
- Python 3.x
- PyTorch (>=1.9.0)
- NumPy (>=1.21.0)
- PyYAML (>=5.4.1)
- scipy (for TGCN graph_conv.py - **MISSING from requirements.txt**)

**Status:**
- ✅ `requirements.txt` exists with torch, numpy, pyyaml
- ❌ **MISSING**: `scipy` dependency (used in `models/tgcn/graph_conv.py` line 2)
- ⚠️ **AMBIGUOUS**: Python version not specified

**Validation Commands:**
```powershell
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check NumPy installation
python -c "import numpy; print(f'NumPy {numpy.__version__}')"

# Check PyYAML installation
python -c "import yaml; print('PyYAML OK')"

# Check scipy (CRITICAL - currently missing)
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
```

---

### b) Data (dataset availability, shapes, adjacency)

**Required Components:**
- Synthetic dataset generator (`datasets/synthetic.py`)
- Data loader with train/val/test splits (`datasets/loader.py`)
- Data transforms with scaling (`datasets/transforms.py`)
- Graph adjacency matrix builder (`graphs/adjacency.py`)

**Status:**
- ✅ Synthetic dataset generator exists and implemented
- ✅ Data loader creates train/val/test splits
- ✅ Scaler fits on training data only
- ✅ Graph builder creates synthetic adjacency
- ⚠️ **AMBIGUOUS**: `num_samples=1000` is hardcoded in `run_single.py` line 54 (not configurable)
- ✅ Data shapes match contract: `x [B, Tin, N, F]`, `y [B, Tout, N, 1]`

**Validation Commands:**
```powershell
# Test dataset generation
python -c "from datasets.synthetic import generate_synthetic_data; x, y = generate_synthetic_data(10, 50, 3, 12, 12); print(f'x shape: {x.shape}, y shape: {y.shape}')"

# Test data loader
python -c "from datasets.loader import create_dataloaders; train, val, test, transform = create_dataloaders(100, 50, 3, 12, 12, 0.7, 0.15, 0.15, 32); print(f'Train batches: {len(train)}, Val batches: {len(val)}, Test batches: {len(test)}')"

# Test graph builder
python -c "from graphs.adjacency import build_graph; adj = build_graph(50, 'synthetic', True, 'symmetric', 42); print(f'Adj shape: {adj.shape}, dtype: {adj.dtype}')"
```

---

### c) Model (wrapper behavior, input/output contract)

**Required Components:**
- TGCN wrapper implements benchmark interface
- Input: `x [B, Tin, N, F]`, `adj [N, N]`
- Output: `y_hat [B, Tout, N, 1]`
- Model can be instantiated via `create_model('tgcn', cfg)`

**Status:**
- ✅ TGCN wrapper exists (`models/tgcn/wrapper.py`)
- ✅ Wrapper implements `forward(x, adj)` signature
- ✅ Feature projection handles F > 1 case
- ✅ Autoregressive rollout for multi-step forecasting
- ✅ Model registered in `models/registry.py`
- ⚠️ **POTENTIAL ISSUE**: Lazy initialization of TGCN in wrapper (line 75-79) - TGCN created on first forward pass
- ⚠️ **POTENTIAL ISSUE**: Wrapper converts adj to numpy (line 77) - device handling needs verification
- ✅ Output shape matches contract: `[B, Tout, N, 1]`

**Validation Commands:**
```powershell
# Test model creation
python -c "from models.registry import create_model; cfg = {'num_nodes': 10, 'num_features': 3, 'seq_len': 12, 'pred_len': 12, 'hidden_dim': 64}; model = create_model('tgcn', cfg); print(f'Model: {type(model).__name__}')"

# Test model forward (requires torch tensor setup)
python -c "import torch; from models.registry import create_model; cfg = {'num_nodes': 5, 'num_features': 3, 'seq_len': 12, 'pred_len': 12, 'hidden_dim': 64}; model = create_model('tgcn', cfg); x = torch.randn(2, 12, 5, 3); adj = torch.randn(5, 5); y = model(x, adj); print(f'Output shape: {y.shape}')"
```

---

### d) Training Loop (trainer, optimizer, loss)

**Required Components:**
- Trainer class (`engine/trainer.py`)
- Optimizer (Adam) configuration
- Loss function (MSE)
- Training loop with validation
- Early stopping
- Gradient clipping
- Checkpointing

**Status:**
- ✅ Trainer class fully implemented
- ✅ Optimizer (Adam) created in `run_single.py` lines 101-105
- ✅ Loss function (MSE) created in `run_single.py` line 106
- ✅ Training loop with epoch iteration
- ✅ Validation loop with metrics
- ✅ Early stopping implemented
- ✅ Gradient clipping implemented
- ✅ Checkpoint saving/loading implemented
- ✅ Trainer receives `adj` and passes to model correctly

**Validation Commands:**
```powershell
# Test trainer import
python -c "from engine.trainer import Trainer; print('Trainer import OK')"

# Test trainer instantiation (requires model, loaders, etc.)
# This would be tested in full run_single.py execution
```

---

### e) Evaluation (metrics, inverse scaling)

**Required Components:**
- Evaluator class (`engine/evaluator.py`)
- Metrics: MAE, RMSE, MAPE (`utils/metrics.py`)
- Inverse scaling of predictions (`utils/scaler.py`)
- Results saving (JSON and text)

**Status:**
- ✅ Evaluator class fully implemented
- ✅ Metrics (MAE, RMSE, MAPE) implemented
- ✅ Inverse scaling implemented in evaluator (line 49-50)
- ✅ Results saved to JSON and text files
- ✅ Scaler inverse_transform implemented
- ✅ Metrics computed on inverse-scaled data

**Validation Commands:**
```powershell
# Test metrics
python -c "import torch; from utils.metrics import mae, rmse, mape; y_true = torch.randn(10, 12, 50, 1); y_pred = torch.randn(10, 12, 50, 1); print(f'MAE: {mae(y_true, y_pred):.4f}, RMSE: {rmse(y_true, y_pred):.4f}, MAPE: {mape(y_true, y_pred):.4f}')"

# Test scaler inverse transform
python -c "import torch; from utils.scaler import StandardScaler; scaler = StandardScaler(); data = torch.randn(100, 12, 50, 3); scaler.fit(data); scaled = scaler.transform(data); restored = scaler.inverse_transform(scaled); print(f'Restored shape: {restored.shape}, Match: {torch.allclose(data, restored, atol=1e-5)}')"
```

---

## 2. Current Project Status

### ✅ Satisfied Prerequisites

1. **Data Pipeline**: Complete
   - Synthetic dataset generator
   - Data loaders with splits
   - Scaling infrastructure
   - Graph builder

2. **Model Infrastructure**: Complete
   - TGCN wrapper implemented
   - Model registry functional
   - Interface contract satisfied

3. **Training Infrastructure**: Complete
   - Trainer with full training loop
   - Optimizer and loss setup
   - Early stopping and checkpointing

4. **Evaluation Infrastructure**: Complete
   - Evaluator with metrics
   - Inverse scaling
   - Results saving

5. **Configuration**: Mostly Complete
   - Base config exists
   - Model config exists
   - All required parameters present

### ❌ Missing Prerequisites

1. **Environment Dependency**: 
   - `scipy` not in `requirements.txt` (used by `models/tgcn/graph_conv.py`)

2. **Configuration Issue**:
   - `num_samples` hardcoded to 1000 in `run_single.py` line 54 (not in config)

### ⚠️ Ambiguous / Needs Confirmation

1. **Model Initialization**:
   - TGCN lazy initialization in wrapper (first forward pass) - may cause issues if model state needs to be on device before training
   - Device handling when converting adj to numpy and back

2. **Graph Normalization**:
   - `build_graph()` normalizes adjacency (line 98 in `graphs/adjacency.py`)
   - But TGCN wrapper calls `calculate_laplacian_with_self_loop()` which also adds self-loops and normalizes
   - Potential double normalization issue

3. **Config Merging**:
   - `run_single.py` merges model config from base config (lines 87-88)
   - But `configs/models/tgcn.yaml` has nested `model:` key
   - Need to verify config structure matches expected format

---

## 3. Minimal Next Steps

### MUST DO (Blocking)

1. **Add scipy to requirements.txt**
   - **Action**: Add `scipy>=1.7.0` to `requirements.txt`
   - **Reason**: Required by `models/tgcn/graph_conv.py` line 2
   - **Validation**: `python -c "import scipy; print('OK')"`

2. **Verify Config Structure**
   - **Action**: Check if `configs/models/tgcn.yaml` structure matches what `run_single.py` expects
   - **Current**: `tgcn.yaml` has `model: { type: "tgcn", hidden_dim: 64 }`
   - **Expected**: `run_single.py` line 87-88 expects `config["model"]` to exist
   - **Issue**: Base config may not have `model:` key, so merge might fail
   - **Validation**: Test config loading and merging

3. **Test Model Forward Pass**
   - **Action**: Verify TGCN wrapper forward pass works with actual tensors
   - **Reason**: Lazy initialization and device handling need verification
   - **Validation**: Run minimal forward pass test

4. **Verify Graph Normalization**
   - **Action**: Check if double normalization occurs (build_graph normalizes, then calculate_laplacian_with_self_loop normalizes again)
   - **Reason**: May affect model performance
   - **Validation**: Inspect adj values before/after each normalization step

### NICE TO HAVE (Non-Blocking)

1. **Make num_samples Configurable**
   - **Action**: Move `num_samples=1000` from hardcoded to config
   - **Reason**: Better flexibility
   - **Priority**: Low (synthetic data, can adjust later)

2. **Add Model Device Verification**
   - **Action**: Add explicit device checks in wrapper initialization
   - **Reason**: Ensure model parameters on correct device
   - **Priority**: Medium (may cause runtime errors if wrong)

3. **Add Input Shape Validation**
   - **Action**: Add assertions in wrapper forward to validate input shapes
   - **Reason**: Better error messages
   - **Priority**: Low (debugging aid)

4. **Document Graph Normalization Strategy**
   - **Action**: Clarify whether double normalization is intentional
   - **Reason**: Code clarity
   - **Priority**: Medium (affects correctness)

---

## 4. Validation Commands (Windows PowerShell)

### Environment Check
```powershell
# Full environment validation
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import yaml; print('PyYAML: OK')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"  # Will fail if not installed
```

### Data Pipeline Check
```powershell
# Test dataset generation
python -c "from datasets.synthetic import generate_synthetic_data; x, y = generate_synthetic_data(10, 50, 3, 12, 12); print(f'✓ Dataset: x{x.shape}, y{y.shape}')"

# Test data loader
python -c "from datasets.loader import create_dataloaders; train, val, test, t = create_dataloaders(100, 50, 3, 12, 12, 0.7, 0.15, 0.15, 32); print(f'✓ Loaders: train={len(train)}, val={len(val)}, test={len(test)}')"

# Test graph builder
python -c "from graphs.adjacency import build_graph; adj = build_graph(50, 'synthetic', True, 'symmetric', 42); print(f'✓ Graph: {adj.shape}, min={adj.min():.3f}, max={adj.max():.3f}')"
```

### Model Check
```powershell
# Test model creation
python -c "from models.registry import create_model; cfg = {'num_nodes': 10, 'num_features': 3, 'seq_len': 12, 'pred_len': 12, 'hidden_dim': 64}; m = create_model('tgcn', cfg); print(f'✓ Model created: {type(m).__name__}')"

# Test model forward (minimal)
python -c "import torch; from models.registry import create_model; cfg = {'num_nodes': 5, 'num_features': 3, 'seq_len': 12, 'pred_len': 12, 'hidden_dim': 64}; model = create_model('tgcn', cfg); x = torch.randn(2, 12, 5, 3); adj = torch.randn(5, 5); y = model(x, adj); print(f'✓ Forward: input{x.shape} -> output{y.shape}')"
```

### Config Check
```powershell
# Test config loading
python -c "from utils.io import load_config; base = load_config('configs/base.yaml'); tgcn = load_config('configs/models/tgcn.yaml'); print(f'✓ Base config keys: {list(base.keys())}'); print(f'✓ TGCN config: {tgcn}')"

# Test config merging
python -c "from utils.io import load_config, merge_configs; base = load_config('configs/base.yaml'); tgcn = load_config('configs/models/tgcn.yaml'); merged = merge_configs(base, tgcn); print(f'✓ Merged model config: {merged.get(\"model\", \"NOT FOUND\")}')"
```

### Full Pipeline Check (Dry Run)
```powershell
# Test full script with minimal epochs (will fail if dependencies missing)
python scripts\run_single.py --config configs\base.yaml --model tgcn --dataset synthetic
# Note: This will actually start training - use Ctrl+C to stop after first epoch
```

### Metrics Check
```powershell
# Test metrics computation
python -c "import torch; from utils.metrics import evaluate; y_true = torch.randn(10, 12, 50, 1); y_pred = torch.randn(10, 12, 50, 1); m = evaluate(y_true, y_pred); print(f'✓ Metrics: {m}')"
```

---

## Summary Checklist

### Critical (Must Fix Before Training)
- [ ] Add `scipy` to `requirements.txt`
- [ ] Verify config structure matches expected format
- [ ] Test model forward pass with actual tensors
- [ ] Verify graph normalization doesn't double-normalize

### Important (Should Verify)
- [ ] Test full pipeline with minimal epochs
- [ ] Verify device handling in model wrapper
- [ ] Check that model parameters are on correct device

### Optional (Nice to Have)
- [ ] Make `num_samples` configurable
- [ ] Add input shape validation
- [ ] Document normalization strategy

---

## Expected Issues to Watch For

1. **Import Error**: `ModuleNotFoundError: No module named 'scipy'` - Fix: Add scipy to requirements
2. **Config KeyError**: If `config["model"]` doesn't exist - Fix: Ensure config merging works
3. **Device Mismatch**: Model on CPU but data on GPU - Fix: Verify device handling in wrapper
4. **Graph Normalization**: Double normalization may affect results - Fix: Review normalization strategy
5. **Lazy Initialization**: Model not initialized before optimizer creation - Fix: May need to call forward once or initialize explicitly

