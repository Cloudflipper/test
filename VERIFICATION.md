# Verification Checklist

This document provides commands to verify the benchmark skeleton is working correctly.

## Prerequisites

- Python environment with required packages (torch, numpy, pyyaml)
- Windows PowerShell

## Verification Commands

### 1. Test Empty Models Suite

Should exit cleanly with code 0 and print "No models specified":

```powershell
python scripts\run_suite.py --suite configs\suite_10.yaml
```

**Expected output:**
```
No models specified in suite configuration.
INFO - No models specified in suite configuration.
```

**Expected exit code:** 0

### 2. Test Non-Integrated Model

Should print friendly message and exit with code 2:

```powershell
python scripts\run_single.py --config configs\base.yaml --model foo --dataset synthetic
```

**Expected output:**
```
Model 'foo' is not integrated yet. Provide source code to integrate.
```

**Expected exit code:** 2

## Repository Structure Verification

Verify the following structure exists:

```
models/
  ├── README.md          # Integration contract
  ├── registry.py        # Factory that raises NotImplementedError
  └── (no __init__.py with model imports)

configs/
  ├── base.yaml
  ├── suite_10.yaml     # Should have models: []
  └── models/
      └── _template.yaml # Template for new models

scripts/
  ├── run_single.py     # Handles NotImplementedError gracefully
  └── run_suite.py      # Handles empty models list
```

## Notes

- NumPy compatibility warnings may appear but do not affect functionality
- All model implementations should be removed
- The registry factory should raise NotImplementedError for any model name
- Scripts should catch NotImplementedError and exit gracefully
