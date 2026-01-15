#!/usr/bin/env python3
"""
Preprocess all HWDB2.x DGRL files from Google Drive folders.
Combines HWDB2.0, 2.1, 2.2 into a single dataset.

Usage in Colab:
    !python preprocess_all_hwdb.py \
        --drive_root /content/drive/MyDrive/handwritten-chinese-ocr-samples \
        --output_dir data/hwdb_full
"""

import os
import struct
import argparse
from PIL import Image
import io
import random
from collections import defaultdict


def parse_dgrl_file(dgrl_path):
    """Parse a single DGRL file and yield (image_bytes, label) tuples."""
    with open(dgrl_path, 'rb') as f:
        # Read header
        header_size = struct.unpack('<I', f.read(4))[0]
        format_code = f.read(8).rstrip(b'\x00').decode('ascii', errors='ignore')
        illustration = f.read(header_size - 62).rstrip(b'\x00').decode('gb18030', errors='ignore')
        
        # Read metadata
        code_type = f.read(20).rstrip(b'\x00').decode('ascii', errors='ignore')
        code_length = struct.unpack('<H', f.read(2))[0]
        bits_per_pixel = struct.unpack('<H', f.read(2))[0]
        
        # Skip to line count
        f.seek(header_size)
        
        # Read number of lines in this page
        num_lines = struct.unpack('<I', f.read(4))[0]
        
        for line_idx in range(num_lines):
            try:
                # Read line header
                line_header_size = struct.unpack('<I', f.read(4))[0]
                if line_header_size == 0:
                    break
                    
                # Read label (in GB18030)
                label_bytes = f.read(line_header_size - 4)
                # Find null terminator
                null_pos = label_bytes.find(b'\x00')
                if null_pos != -1:
                    label_bytes = label_bytes[:null_pos]
                label = label_bytes.decode('gb18030', errors='ignore').strip()
                
                if not label:
                    continue
                
                # Read line image dimensions
                line_height = struct.unpack('<I', f.read(4))[0]
                line_width = struct.unpack('<I', f.read(4))[0]
                
                if line_height == 0 or line_width == 0 or line_height > 5000 or line_width > 5000:
                    continue
                
                # Read image data
                image_size = line_height * line_width
                image_data = f.read(image_size)
                
                if len(image_data) != image_size:
                    break
                
                yield image_data, line_width, line_height, label
                
                # Skip padding (0xFF bytes between lines)
                while True:
                    peek = f.read(1)
                    if not peek:
                        break
                    if peek != b'\xff':
                        f.seek(-1, 1)
                        break
                        
            except Exception as e:
                print(f"  Warning: Error parsing line {line_idx}: {e}")
                break


def process_dgrl_folder(folder_path, output_dir, split, all_chars, sample_count):
    """Process all DGRL files in a folder."""
    if not os.path.exists(folder_path):
        print(f"  Folder not found: {folder_path}")
        return sample_count
    
    dgrl_files = [f for f in os.listdir(folder_path) if f.endswith('.dgrl')]
    print(f"  Found {len(dgrl_files)} DGRL files in {os.path.basename(folder_path)}")
    
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    img_gt_list = []
    
    for dgrl_file in sorted(dgrl_files):
        dgrl_path = os.path.join(folder_path, dgrl_file)
        page_name = os.path.splitext(dgrl_file)[0]
        
        try:
            for line_idx, (image_data, width, height, label) in enumerate(parse_dgrl_file(dgrl_path)):
                # Save image
                img_name = f"{page_name}_L{line_idx:03d}.png"
                img_path = os.path.join(split_dir, img_name)
                
                try:
                    img = Image.frombytes('L', (width, height), image_data)
                    img.save(img_path)
                    
                    img_gt_list.append((img_name, label))
                    sample_count += 1
                    
                    # Collect characters
                    for char in label:
                        all_chars.add(char)
                        
                except Exception as e:
                    print(f"    Warning: Failed to save {img_name}: {e}")
                    
        except Exception as e:
            print(f"    Warning: Failed to parse {dgrl_file}: {e}")
    
    return sample_count, img_gt_list


def main():
    parser = argparse.ArgumentParser(description='Preprocess all HWDB2.x DGRL files')
    parser.add_argument('--drive_root', type=str, required=True,
                        help='Root path of the project in Google Drive')
    parser.add_argument('--output_dir', type=str, default='data/hwdb_full',
                        help='Output directory for processed data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    args = parser.parse_args()
    
    drive_root = args.drive_root
    output_dir = os.path.join(drive_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HWDB2.x Full Dataset Preprocessing")
    print("=" * 60)
    
    # Define folder mappings
    train_folders = [
        os.path.join(drive_root, 'HWDB2.0Train'),
        os.path.join(drive_root, 'HWDB2.1Train'),
        os.path.join(drive_root, 'HWDB2.2Train'),
    ]
    
    test_folders = [
        os.path.join(drive_root, 'HWDB2.0Test'),
        os.path.join(drive_root, 'HWDB2.1Test'),
        os.path.join(drive_root, 'HWDB2.2Test'),
    ]
    
    all_chars = set()
    
    # Process training data
    print("\n[1/2] Processing Training Data...")
    train_samples = []
    train_count = 0
    
    for folder in train_folders:
        print(f"\n  Processing: {os.path.basename(folder)}")
        train_count, samples = process_dgrl_folder(folder, output_dir, 'train_all', all_chars, train_count)
        train_samples.extend(samples)
    
    print(f"\n  Total training samples collected: {len(train_samples)}")
    
    # Split into train and validation
    random.seed(42)
    random.shuffle(train_samples)
    
    val_size = int(len(train_samples) * args.val_ratio)
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]
    
    # Move validation samples to val folder
    train_all_dir = os.path.join(output_dir, 'train_all')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"\n  Splitting: {len(train_samples)} train, {len(val_samples)} val")
    
    # Move files
    for img_name, label in train_samples:
        src = os.path.join(train_all_dir, img_name)
        dst = os.path.join(train_dir, img_name)
        if os.path.exists(src):
            os.rename(src, dst)
    
    for img_name, label in val_samples:
        src = os.path.join(train_all_dir, img_name)
        dst = os.path.join(val_dir, img_name)
        if os.path.exists(src):
            os.rename(src, dst)
    
    # Clean up temp folder
    if os.path.exists(train_all_dir):
        os.rmdir(train_all_dir)
    
    # Process test data
    print("\n[2/2] Processing Test Data...")
    test_samples = []
    test_count = 0
    
    for folder in test_folders:
        print(f"\n  Processing: {os.path.basename(folder)}")
        test_count, samples = process_dgrl_folder(folder, output_dir, 'test', all_chars, test_count)
        test_samples.extend(samples)
    
    print(f"\n  Total test samples: {len(test_samples)}")
    
    # Save image-groundtruth files
    print("\n[3/3] Saving metadata files...")
    
    with open(os.path.join(output_dir, 'train_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in train_samples:
            f.write(f"{img_name},{label}\n")
    
    with open(os.path.join(output_dir, 'val_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in val_samples:
            f.write(f"{img_name},{label}\n")
    
    with open(os.path.join(output_dir, 'test_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in test_samples:
            f.write(f"{img_name},{label}\n")
    
    # Save chars list
    # Sort characters for consistency
    chars_list = sorted(list(all_chars))
    with open(os.path.join(output_dir, 'chars_list.txt'), 'w', encoding='utf-8') as f:
        for char in chars_list:
            f.write(char + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}")
    print(f"Character vocabulary: {len(chars_list)}")
    print("=" * 60)
    
    # Verify sample labels
    print("\nSample labels (first 5 training samples):")
    for img_name, label in train_samples[:5]:
        print(f"  {img_name}: {label[:30]}{'...' if len(label) > 30 else ''}")


if __name__ == '__main__':
    main()
