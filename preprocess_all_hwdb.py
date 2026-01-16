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
import numpy as np
from PIL import Image
import argparse
import random
from multiprocessing import Pool, cpu_count
import cv2
import shutil
import time
from pathlib import Path


def _find_next_line(data, start_pos, debug=False):
    """Heuristic resync: scan forward to find plausible next line header.
    Looks for num_chars in [1,150] followed by valid dimensions.
    """
    scan_limit = min(len(data) - 12, start_pos + 500000)  # Increased scan limit

    if debug:
        print(f"    Scanning from {start_pos} to {scan_limit} (range: {scan_limit - start_pos} bytes)")

    for pos in range(start_pos, scan_limit):
        if pos + 4 > len(data):
            break

        num_chars = struct.unpack('<I', data[pos:pos+4])[0]
        if 1 <= num_chars <= 150:
            txt_end = pos + 4 + num_chars * 2
            if txt_end + 8 <= len(data):
                height = struct.unpack('<I', data[txt_end:txt_end+4])[0]
                width = struct.unpack('<I', data[txt_end+4:txt_end+8])[0]
                if 1 <= height <= 4000 and 1 <= width <= 8000:
                    if debug:
                        print(f"    Resync: num_chars={num_chars} height={height} width={width} at pos={pos}")
                    return pos

    if debug:
        print(f"    No valid line header found in scan range")
    return None


def parse_dgrl_file(filepath, debug=False):
    """Parse a single DGRL file and extract text lines with images.
    Uses a resync heuristic to handle trailing bbox/meta data.
    """
    lines_data = []
    
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        
        if len(data) < 85:
            return lines_data
        
        # Header: 73 bytes
        # Offset 73: page_width (4 bytes)
        # Offset 77: page_height (4 bytes)  
        # Offset 81: num_lines (4 bytes)
        # Offset 85: line records start
        
        pos = 73
        page_w = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        page_h = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        num_lines = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        
        if debug:
            print(f"  Page: {page_w}x{page_h}, {num_lines} lines")
        
        if num_lines > 50 or num_lines == 0:  # Sanity check
            return lines_data
        
        for line_idx in range(num_lines):
            if pos + 4 > len(data):
                break
            
            # Number of characters
            num_chars = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4

            if num_chars > 200 or num_chars == 0:  # Sanity check
                if debug:
                    print(f"  Line {line_idx}: Invalid num_chars={num_chars}, stopping parse")
                break
            
            # Read GB codes (2 bytes each)
            text = ''
            for _ in range(num_chars):
                if pos + 2 > len(data):
                    break
                code = data[pos:pos+2]
                pos += 2
                try:
                    char = code.decode('gb18030')
                    text += char
                except:
                    pass
            
            if pos + 8 > len(data):
                break
            
            # Image dimensions (height first, then width)
            height = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4
            width = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4

            # Sanity check dimensions
            if height > 4000 or width > 8000 or height == 0 or width == 0:
                if debug:
                    print(f"  Line {line_idx}: Invalid dimensions h={height} w={width}, stopping parse")
                break
            
            # Read pixel data
            img_size = height * width
            if pos + img_size > len(data):
                break
            
            img_array = np.frombuffer(data[pos:pos+img_size], dtype=np.uint8)
            img_array = img_array.reshape(height, width)
            pos += img_size
            
            # Skip character bounding boxes (8 bytes per char: x, y, w, h each 2 bytes)
            bbox_size = num_chars * 8
            pos += bbox_size

            # KEY FIX: After bbox data, there's variable-length trailing metadata.
            # Use the _find_next_line helper to scan forward and find the next line header
            if debug:
                print(f"  Line {line_idx}: Looking for next line starting at pos {pos}")

            next_pos = _find_next_line(data, pos, debug)

            if next_pos is not None:
                if debug:
                    print(f"  Line {line_idx}: Found next line at pos {next_pos} (skipped {next_pos - pos} bytes)")
                pos = next_pos
            else:
                # No more lines found - save this line and exit loop
                if debug:
                    print(f"  Line {line_idx}: No more lines found, stopping after line {line_idx + 1}")
                if text:
                    lines_data.append({
                        'image': img_array,
                        'text': text,
                        'height': height,
                        'width': width
                    })
                break
            
            if text:
                lines_data.append({
                    'image': img_array,
                    'text': text,
                    'height': height,
                    'width': width
                })
                
    except Exception as e:
        if debug:
            print(f"  Error parsing {filepath}: {e}")
    
    return lines_data


def process_single_dgrl(args):
    """Worker function to process a single DGRL file."""
    dgrl_path, output_dir, split, target_height = args

    page_name = os.path.splitext(os.path.basename(dgrl_path))[0]
    split_dir = os.path.join(output_dir, split)

    lines = parse_dgrl_file(dgrl_path)

    img_gt_list = []
    chars_set = set()

    for line_idx, line_data in enumerate(lines):
        try:
            # Resize to target height
            img = line_data['image']
            h, w = img.shape
            ratio = target_height / h
            new_w = max(1, int(w * ratio))

            # KEY FIX: Ensure minimum width for CTC (text_len * 20 pixels per char)
            text_len = len(line_data['text'])
            min_width = text_len * 20  # 20 pixels per character minimum
            if new_w < min_width:
                new_w = min_width

            # Use cv2 instead of PIL (10x faster)
            resized_img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

            # Save image with no compression for faster writes to local storage
            img_name = f"{page_name}_L{line_idx:03d}.png"
            img_path = os.path.join(split_dir, img_name)
            cv2.imwrite(img_path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            img_gt_list.append((img_name, line_data['text']))

            # Collect characters
            for char in line_data['text']:
                chars_set.add(char)

        except Exception as e:
            pass  # Skip problematic lines silently

    return img_gt_list, chars_set, len(lines) > 0


def process_dgrl_folder(folder_path, output_dir, split, all_chars, target_height=128, num_workers=4):
    """Process all DGRL files in a folder with multiprocessing."""
    if not os.path.exists(folder_path):
        print(f"  Folder not found: {folder_path}")
        return []

    dgrl_files = [f for f in os.listdir(folder_path) if f.endswith('.dgrl')]
    print(f"  Found {len(dgrl_files)} DGRL files in {os.path.basename(folder_path)}")

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    args_list = [
        (os.path.join(folder_path, dgrl_file), output_dir, split, target_height)
        for dgrl_file in sorted(dgrl_files)
    ]

    img_gt_list = []
    processed_files = 0
    total_lines = 0

    # Process files in parallel with progress tracking
    print(f"    Processing with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(pool.imap(process_single_dgrl, args_list))

    # Merge results
    for file_img_gt_list, chars_set, has_lines in results:
        img_gt_list.extend(file_img_gt_list)
        all_chars.update(chars_set)
        total_lines += len(file_img_gt_list)
        if has_lines:
            processed_files += 1

    print(f"    Processed {processed_files}/{len(dgrl_files)} files, {total_lines} lines")
    return img_gt_list


def main():
    parser = argparse.ArgumentParser(description='Preprocess all HWDB2.x DGRL files')
    parser.add_argument('--drive_root', type=str, required=True,
                        help='Root path of the project in Google Drive')
    parser.add_argument('--output_dir', type=str, default='data/hwdb_full',
                        help='Output directory for processed data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--use_local_storage', type=bool, default=True,
                        help='Use local /tmp for faster processing (default: True)')
    args = parser.parse_args()

    drive_root = args.drive_root
    final_output_dir = os.path.join(drive_root, args.output_dir)

    # OPTIMIZATION: Use local /tmp storage for fast processing
    if args.use_local_storage:
        # Check available disk space
        disk_usage = shutil.disk_usage('/tmp')
        required_space = 2_000_000_000  # 2GB safety margin
        available_gb = disk_usage.free / 1e9

        if disk_usage.free < required_space:
            print(f"WARNING: Low disk space ({available_gb:.1f}GB free, recommend 2GB+)")
            print("Falling back to direct Google Drive writes...")
            output_dir = final_output_dir
            use_local = False
        else:
            output_dir = '/tmp/hwdb_processing'
            use_local = True
            print(f"Using local storage: {output_dir} ({available_gb:.1f}GB available)")
    else:
        output_dir = final_output_dir
        use_local = False

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HWDB2.x Full Dataset Preprocessing (OPTIMIZED)")
    print("=" * 60)
    if use_local:
        print(f"Local processing dir: {output_dir}")
        print(f"Final destination: {final_output_dir}")
    else:
        print(f"Output directory: {output_dir}")
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
    print("\n[1/3] Processing Training Data...")
    all_train_samples = []
    
    for folder in train_folders:
        if os.path.exists(folder):
            print(f"\n  Processing: {os.path.basename(folder)}")
            samples = process_dgrl_folder(folder, output_dir, 'train_temp', all_chars, num_workers=args.workers)
            all_train_samples.extend(samples)
        else:
            print(f"  Skipping (not found): {folder}")
    
    print(f"\n  Total training samples collected: {len(all_train_samples)}")
    
    # Process test data
    print("\n[2/3] Processing Test Data...")
    all_test_samples = []
    
    for folder in test_folders:
        if os.path.exists(folder):
            print(f"\n  Processing: {os.path.basename(folder)}")
            samples = process_dgrl_folder(folder, output_dir, 'test', all_chars, num_workers=args.workers)
            all_test_samples.extend(samples)
        else:
            print(f"  Skipping (not found): {folder}")
    
    print(f"\n  Total test samples: {len(all_test_samples)}")
    
    # Split train into train and validation
    print("\n[3/3] Splitting train/val and saving metadata...")
    
    random.seed(42)
    random.shuffle(all_train_samples)
    
    val_size = int(len(all_train_samples) * args.val_ratio)
    val_samples = all_train_samples[:val_size]
    train_samples = all_train_samples[val_size:]
    
    # Create train and val directories, move files
    train_temp_dir = os.path.join(output_dir, 'train_temp')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Move training files
    for img_name, label in train_samples:
        src = os.path.join(train_temp_dir, img_name)
        dst = os.path.join(train_dir, img_name)
        if os.path.exists(src):
            os.rename(src, dst)
    
    # Move validation files
    for img_name, label in val_samples:
        src = os.path.join(train_temp_dir, img_name)
        dst = os.path.join(val_dir, img_name)
        if os.path.exists(src):
            os.rename(src, dst)
    
    # Clean up temp folder
    try:
        if os.path.exists(train_temp_dir):
            os.rmdir(train_temp_dir)
    except:
        pass
    
    # Save image-groundtruth files
    with open(os.path.join(output_dir, 'train_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in train_samples:
            f.write(f"{img_name},{label}\n")
    
    with open(os.path.join(output_dir, 'val_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in val_samples:
            f.write(f"{img_name},{label}\n")
    
    with open(os.path.join(output_dir, 'test_img_id_gt.txt'), 'w', encoding='utf-8') as f:
        for img_name, label in all_test_samples:
            f.write(f"{img_name},{label}\n")
    
    # Save chars list
    chars_list = sorted(list(all_chars))
    with open(os.path.join(output_dir, 'chars_list.txt'), 'w', encoding='utf-8') as f:
        for char in chars_list:
            f.write(char + '\n')

    # OPTIMIZATION: Bulk copy from local storage to Google Drive
    if use_local and output_dir != final_output_dir:
        print("\n" + "=" * 60)
        print("Copying processed data to Google Drive...")
        print("=" * 60)

        print(f"Source: {output_dir}")
        print(f"Destination: {final_output_dir}")

        start_time = time.time()

        # Bulk copy entire directory tree
        shutil.copytree(output_dir, final_output_dir, dirs_exist_ok=True)

        copy_time = time.time() - start_time
        print(f"Copy completed in {copy_time:.1f} seconds")

        # Verify copy succeeded
        local_png_count = len(list(Path(output_dir).rglob('*.png')))
        drive_png_count = len(list(Path(final_output_dir).rglob('*.png')))
        local_txt_count = len(list(Path(output_dir).rglob('*.txt')))
        drive_txt_count = len(list(Path(final_output_dir).rglob('*.txt')))

        print(f"\nVerification:")
        print(f"  PNG files: {local_png_count} local -> {drive_png_count} Drive")
        print(f"  TXT files: {local_txt_count} local -> {drive_txt_count} Drive")

        if local_png_count != drive_png_count or local_txt_count != drive_txt_count:
            print("\nWARNING: File count mismatch! Copy may have failed.")
            print(f"Keeping local files at: {output_dir}")
        else:
            print("\n✅ Copy verified successfully!")
            print("Cleaning up local files...")
            shutil.rmtree(output_dir)
            print(f"✅ Local files removed: {output_dir}")

        # Update output_dir for final summary
        output_dir = final_output_dir

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(all_test_samples)}")
    print(f"Total samples: {len(train_samples) + len(val_samples) + len(all_test_samples)}")
    print(f"Character vocabulary: {len(chars_list)}")
    print("=" * 60)
    
    # Show sample labels
    print("\nSample training labels:")
    for img_name, label in train_samples[:3]:
        print(f"  {img_name}: {label[:40]}{'...' if len(label) > 40 else ''}")


if __name__ == '__main__':
    main()
