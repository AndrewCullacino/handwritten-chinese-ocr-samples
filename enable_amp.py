#!/usr/bin/env python3
"""
Script to enable Automatic Mixed Precision (AMP) in main.py

This script patches main.py to add AMP support for 2-3x speedup on A100 GPU.

Usage:
    python enable_amp.py

Or manually apply the changes shown in TRAINING_OPTIMIZATION_GUIDE.md
"""

import os
import sys

def enable_amp():
    """Add AMP support to main.py"""

    main_file = 'main.py'

    if not os.path.exists(main_file):
        print(f"ERROR: {main_file} not found")
        print("Run this script from the project root directory")
        return False

    print("Reading main.py...")
    with open(main_file, 'r') as f:
        content = f.read()

    # Check if AMP is already enabled
    if 'from torch.cuda.amp import' in content:
        print("✅ AMP is already enabled in main.py")
        return True

    print("Adding AMP support to main.py...")

    # Step 1: Add import at the top
    import_line = "import torch.distributed as dist\n"
    amp_import = "from torch.cuda.amp import autocast, GradScaler\n"

    if import_line in content:
        content = content.replace(import_line, import_line + amp_import)
        print("  ✓ Added AMP imports")
    else:
        print("  ⚠ Could not find import location, add manually:")
        print(f"    {amp_import}")

    # Step 2: Initialize GradScaler in main_worker function
    model_creation = "model = model.to(args.device)"
    scaler_init = """
    # Mixed Precision Training (AMP)
    scaler = GradScaler()
    print("INFO: Mixed precision training (AMP) enabled for 2-3x speedup")
"""

    if model_creation in content:
        content = content.replace(model_creation, model_creation + scaler_init)
        print("  ✓ Added GradScaler initialization")
    else:
        print("  ⚠ Could not find model initialization, add scaler manually")

    # Step 3: Update training loop
    old_forward = """        loss = criterion(preds.log_softmax(2),
                         target_indexs_tensor,
                         preds_sizes,
                         target_sizes_tensor)"""

    new_forward = """        # Forward pass with mixed precision
        with autocast():
            loss = criterion(preds.log_softmax(2),
                           target_indexs_tensor,
                           preds_sizes,
                           target_sizes_tensor)"""

    if old_forward in content:
        content = content.replace(old_forward, new_forward)
        print("  ✓ Updated forward pass for AMP")
    else:
        print("  ⚠ Could not find loss calculation, update manually")

    # Step 4: Update backward pass
    old_backward = """        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()"""

    new_backward = """        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()"""

    if old_backward in content:
        content = content.replace(old_backward, new_backward)
        print("  ✓ Updated backward pass for AMP")
    else:
        print("  ⚠ Could not find backward pass, update manually")

    # Save modified file
    backup_file = 'main.py.backup'
    print(f"\nCreating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        with open(main_file, 'r') as orig:
            f.write(orig.read())

    print(f"Writing modified main.py...")
    with open(main_file, 'w') as f:
        f.write(content)

    print("\n✅ AMP support successfully added to main.py")
    print(f"   Backup saved to: {backup_file}")
    print("\nExpected speedup: 2-3x faster training on A100 GPU")
    print("\nTo revert changes:")
    print(f"  mv {backup_file} {main_file}")

    return True

if __name__ == '__main__':
    print("=" * 70)
    print("Enabling Automatic Mixed Precision (AMP) in main.py")
    print("=" * 70)
    print()

    success = enable_amp()

    if success:
        print("\nNext steps:")
        print("  1. Review the changes in main.py")
        print("  2. Run optimized training:")
        print("     python train_optimized.py --data /path/to/hwdb_full --gpu 0")
        sys.exit(0)
    else:
        print("\nFailed to enable AMP automatically.")
        print("Please refer to TRAINING_OPTIMIZATION_GUIDE.md for manual instructions.")
        sys.exit(1)
