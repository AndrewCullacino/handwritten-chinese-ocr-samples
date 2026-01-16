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


def process_dgrl_folder(folder_path, output_dir, split, all_chars, target_height=128):
    """Process all DGRL files in a folder."""
    if not os.path.exists(folder_path):
        print(f"  Folder not found: {folder_path}")
        return []
    
    dgrl_files = [f for f in os.listdir(folder_path) if f.endswith('.dgrl')]
    print(f"  Found {len(dgrl_files)} DGRL files in {os.path.basename(folder_path)}")
    
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    img_gt_list = []
    processed_files = 0
    total_lines = 0
    
    for dgrl_file in sorted(dgrl_files):
        dgrl_path = os.path.join(folder_path, dgrl_file)
        page_name = os.path.splitext(dgrl_file)[0]
        
        lines = parse_dgrl_file(dgrl_path)
        
        if lines:
            processed_files += 1
            
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

                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
                
                # Save image
                img_name = f"{page_name}_L{line_idx:03d}.png"
                img_path = os.path.join(split_dir, img_name)
                pil_img.save(img_path)
                
                img_gt_list.append((img_name, line_data['text']))
                total_lines += 1
                
                # Collect characters
                for char in line_data['text']:
                    all_chars.add(char)
                    
            except Exception as e:
                pass  # Skip problematic lines silently
    
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
    print("\n[1/3] Processing Training Data...")
    all_train_samples = []
    
    for folder in train_folders:
        if os.path.exists(folder):
            print(f"\n  Processing: {os.path.basename(folder)}")
            samples = process_dgrl_folder(folder, output_dir, 'train_temp', all_chars)
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
            samples = process_dgrl_folder(folder, output_dir, 'test', all_chars)
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
