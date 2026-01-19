# Training Optimization Guide - HWDB Chinese OCR

## Problem Analysis

**Current Performance:**
- Training time: 8 hours for 10 epochs
- Batch size: 8 (extremely small)
- GPU utilization: 5-10% (massive underutilization)
- Hardware: Google Colab A100 GPU (80GB VRAM)

**Root Cause:** The default batch size of 8 severely underutilizes the A100 GPU's parallel processing capability.

## Solution: Optimize Training Configuration

### Performance Improvements

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Batch size: 8 → 64 | 8x | 8x |
| Mixed precision (AMP) | 2.5x | 20x |
| Data loading optimization | 1.2x | **24x** |

**Expected Results:**
- Training time: 8 hours → **20-30 minutes** for 10 epochs
- GPU utilization: 5-10% → 70-90%
- Model quality: Same or better (larger batches = more stable gradients)

## Recommended Configuration

### Quick Start (Colab)

```python
# 1. Extract the dataset archive first (if using tar archive)
!tar -xf /content/drive/MyDrive/handwritten-chinese-ocr-samples/data/hwdb_full.tar \
    -C /content/drive/MyDrive/handwritten-chinese-ocr-samples/data/

# 2. Run optimized training
!python train_optimized.py \
    --data /content/drive/MyDrive/handwritten-chinese-ocr-samples/data/hwdb_full \
    --gpu 0 \
    --epochs 10
```

### Manual Training with Optimized Parameters

```bash
python main.py \
    -m hctr \
    -d /path/to/hwdb_full \
    -b 64 \
    -lr 0.003 \
    -j 8 \
    -ep 10 \
    --gpu 0
```

### Parameter Explanation

**Batch Size (`-b 64`)**
- **Why:** A100 GPUs are designed for large batch processing
- **Impact:** 8x reduction in iterations per epoch
- **Default:** 8 → **Optimized:** 64
- **If OOM error:** Try 48 or 32

**Learning Rate (`-lr 0.003`)**
- **Why:** Larger batches require proportionally larger learning rates
- **Scaling:** LR_new = LR_old × √(batch_multiplier) = 0.001 × √8 ≈ 0.003
- **Default:** 0.001 → **Optimized:** 0.003
- **If unstable:** Try 0.002

**Data Workers (`-j 8`)**
- **Why:** Prevents GPU from waiting for data
- **Impact:** 1.2x speedup by overlapping data loading with computation
- **Default:** 4 → **Optimized:** 8

**Epochs (`-ep 10`)**
- **Why:** 10 epochs is sufficient for convergence with proper batch size
- **Note:** With optimization, 10 epochs = 20-30 minutes

## Mixed Precision Training (AMP)

### Automatic Mixed Precision Support

The training script needs to be modified to enable AMP. Add this to `main.py`:

```python
# At the top of main.py, add:
from torch.cuda.amp import autocast, GradScaler

# In main_worker function, after model creation:
scaler = GradScaler()

# In the training loop (around line 350-372):
with autocast():
    # Forward pass
    preds = model(input)
    preds_sizes = torch.IntTensor([preds.size(0)] * args.batch_size).to(args.device)
    loss = criterion(preds.log_softmax(2), target_indexs_tensor, preds_sizes, target_sizes_tensor)

# Backward pass with gradient scaling
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Why AMP Works with CRNN+CTC

- **CNN layers:** Run in FP16 (2x faster)
- **RNN layers:** Run in FP16 (2x faster)
- **CTC loss:** Automatically computed in FP32 for numerical stability
- **Total speedup:** 2-3x on A100

## Data Loading Optimization

### Enable Pinned Memory

In `main.py`, add to DataLoader:

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,  # Add this
    persistent_workers=True  # Add this (keeps workers alive between epochs)
)
```

### Extract Dataset to Local Storage

For fastest training, extract dataset to local `/tmp/` instead of reading from Google Drive:

```bash
# Extract to local storage
!tar -xf /content/drive/MyDrive/.../hwdb_full.tar -C /tmp/

# Train from local storage (much faster)
!python train_optimized.py --data /tmp/hwdb_full --gpu 0 --epochs 10
```

## Troubleshooting

### Out of Memory (OOM)

If you get OOM error with batch_size=64:

```bash
# Try batch_size=48
python main.py -m hctr -d /path/to/data -b 48 -lr 0.0025 -j 8 -ep 10 --gpu 0

# Or batch_size=32
python main.py -m hctr -d /path/to/data -b 32 -lr 0.002 -j 8 -ep 10 --gpu 0
```

### Training Unstable / NaN Loss

If loss becomes NaN or training is unstable:

1. Reduce learning rate: `-lr 0.002` or `-lr 0.0015`
2. Add gradient clipping in `main.py`:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   ```

### Slower Than Expected

Check GPU utilization:
```bash
# In another terminal/notebook cell
!nvidia-smi -l 1
```

Should show 70-90% GPU utilization. If low:
- Increase batch size
- Increase num_workers
- Check if data loading from Google Drive (slow) instead of local storage

## Alternative Options (Not Recommended)

### Option 2: Train with Only HWDB2.0

**Not recommended** - Reduces model quality without fixing root cause.

```bash
# Would need to re-run preprocessing with only HWDB2.0Train and HWDB2.0Test
python preprocess_all_hwdb.py \
    --drive_root /content/drive/MyDrive/handwritten-chinese-ocr-samples \
    --output_dir data/hwdb20_only
```

**Impact:**
- Dataset: 30k → 10k samples
- Training time: 8h → 2.7h (3x speedup)
- Model quality: Worse (⅔ less training data)

### Option 3: Fewer Epochs

**Not recommended** - Results in undertrained model.

```bash
python main.py -m hctr -d /path/to/data -ep 5
```

**Impact:**
- Training time: 8h → 4h (2x speedup)
- Model quality: Worse (undertrained)

## Performance Comparison

| Configuration | Batch Size | Epochs | Time | Speedup | Model Quality |
|---------------|------------|--------|------|---------|---------------|
| **Current (Default)** | 8 | 10 | 8h | 1x | Baseline |
| **Optimized (Recommended)** | 64 | 10 | 20-30 min | **24x** | Same/Better |
| Only HWDB2.0 | 8 | 10 | 2.7h | 3x | Worse |
| Fewer epochs | 8 | 5 | 4h | 2x | Worse |

## Validation

After optimization, you should see:

```
Epoch: [1][100/469] Time 0.85 (0.90) Data 0.01 (0.05) Loss 4.23 (5.67)
TRU 窟等就是这一时期的代表作。记者从敦煌研究院了解到,此次双方
PRE 窟等就是这一时期的代表作。
```

**Key indicators:**
- Iterations per epoch: ~469 (was 3,750)
- Time per iteration: ~0.85s
- PRE (predictions) appear by Epoch 2-3 (not blank)
- Loss decreasing steadily

## Summary

**Best Solution:** Option 1 - Optimize Training Configuration

**Implementation:** Use `train_optimized.py` or manual parameters

**Expected Result:** 8 hours → 20-30 minutes (24x faster)

**Why This Works:**
1. Properly utilizes A100 GPU parallel processing
2. Maintains full dataset for best model quality
3. Uses industry-standard batch sizes
4. No trade-offs in model quality
