"""
DGRL Preprocessor for CASIA-HWDB2.x
Converts .dgrl files to PNG images + labels for OCR training.
"""

import os
import struct
import numpy as np
from PIL import Image
import argparse


def parse_dgrl_file(filepath, debug=False):
    """Parse a single DGRL file and extract text lines with images."""
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
            if debug:
                print(f"  SKIP: num_lines={num_lines} out of range")
            return lines_data
        
        for line_idx in range(num_lines):
            if pos + 4 > len(data):
                break
            
            # Number of characters
            num_chars = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4
            
            if debug:
                print(f"  Line {line_idx}: num_chars={num_chars} at pos={pos-4}")
            
            if num_chars > 200 or num_chars == 0:  # Sanity check
                if debug:
                    print(f"  SKIP: num_chars={num_chars} out of range")
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
            
            if debug:
                print(f"    Image: {width}x{height}, text: {text[:20]}...")
            
            # Sanity check dimensions (text lines can be tall, allow up to 2000)
            if height > 2000 or width > 8000 or height == 0 or width == 0:
                if debug:
                    print(f"  SKIP: dimensions {width}x{height} out of range")
                break
            
            # Read pixel data
            img_size = height * width
            if pos + img_size > len(data):
                break
            
            img_array = np.frombuffer(data[pos:pos+img_size], dtype=np.uint8)
            img_array = img_array.reshape(height, width)
            pos += img_size
            
            # Skip line trailing data (bounding box info, ~20 bytes) until we hit 0xFF
            while pos < len(data) and data[pos] != 0xFF:
                pos += 1
            
            # Skip 0xFF padding bytes between lines
            while pos < len(data) and data[pos] == 0xFF:
                pos += 1
            
            if text:
                lines_data.append({
                    'image': img_array,
                    'text': text,
                    'height': height,
                    'width': width
                })
                
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
    
    return lines_data


def process_directory(input_dir, output_dir, phase, target_height=128):
    """Process all DGRL files in a directory."""
    os.makedirs(os.path.join(output_dir, phase), exist_ok=True)
    
    dgrl_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.dgrl')])
    print(f"Found {len(dgrl_files)} DGRL files in {input_dir}")
    
    all_pairs = []
    all_chars = set()
    
    for i, dgrl_file in enumerate(dgrl_files):
        filepath = os.path.join(input_dir, dgrl_file)
        lines = parse_dgrl_file(filepath)
        
        for line_idx, line_data in enumerate(lines):
            # Resize to target height
            img = line_data['image']
            h, w = img.shape
            ratio = target_height / h
            new_w = max(1, int(w * ratio))
            
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
            
            # Save image
            img_name = f"{dgrl_file.replace('.dgrl', '')}_{line_idx:03d}.png"
            pil_img.save(os.path.join(output_dir, phase, img_name))
            
            all_pairs.append((img_name, line_data['text']))
            all_chars.update(line_data['text'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dgrl_files)}, {len(all_pairs)} lines")
    
    # Save ground truth
    gt_file = os.path.join(output_dir, f'{phase}_img_id_gt.txt')
    with open(gt_file, 'w', encoding='utf-8') as f:
        for img_name, text in all_pairs:
            f.write(f'{img_name},{text}\n')
    
    print(f"{phase}: {len(all_pairs)} lines extracted")
    return all_chars


def main():
    parser = argparse.ArgumentParser(description='Preprocess CASIA-HWDB2.x DGRL files')
    parser.add_argument('--train_dir', type=str, default='HWDB2.0Train')
    parser.add_argument('--test_dir', type=str, default='HWDB2.0Test')
    parser.add_argument('--output_dir', type=str, default='data/hwdb2.0')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--target_height', type=int, default=128)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_chars = set()
    
    # Process training data
    if os.path.isdir(args.train_dir):
        print("Processing training data...")
        chars = process_directory(args.train_dir, args.output_dir, 'train', args.target_height)
        all_chars.update(chars)
        
        # Split validation
        train_gt = os.path.join(args.output_dir, 'train_img_id_gt.txt')
        if os.path.exists(train_gt):
            with open(train_gt, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            import random
            random.seed(42)
            random.shuffle(lines)
            
            val_size = int(len(lines) * args.val_split)
            val_lines, train_lines = lines[:val_size], lines[val_size:]
            
            # Move val images
            val_dir = os.path.join(args.output_dir, 'val')
            train_dir = os.path.join(args.output_dir, 'train')
            os.makedirs(val_dir, exist_ok=True)
            
            for line in val_lines:
                img_name = line.strip().split(',')[0]
                src = os.path.join(train_dir, img_name)
                dst = os.path.join(val_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
            
            with open(train_gt, 'w', encoding='utf-8') as f:
                f.writelines(train_lines)
            with open(os.path.join(args.output_dir, 'val_img_id_gt.txt'), 'w', encoding='utf-8') as f:
                f.writelines(val_lines)
            
            print(f"Split: {len(train_lines)} train, {len(val_lines)} val")
    
    # Process test data
    if os.path.isdir(args.test_dir):
        print("Processing test data...")
        chars = process_directory(args.test_dir, args.output_dir, 'test', args.target_height)
        all_chars.update(chars)
    
    # Save char list (merge with existing)
    existing = 'data/handwritten_ctr_data/chars_list.txt'
    if os.path.exists(existing):
        with open(existing, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_chars.add(line.strip())
    
    with open(os.path.join(args.output_dir, 'chars_list.txt'), 'w', encoding='utf-8') as f:
        for char in sorted(all_chars):
            f.write(char + '\n')
    
    print(f"\nDone! {len(all_chars)} unique characters")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
