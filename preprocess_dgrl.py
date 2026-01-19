#!/usr/bin/env python3
"""
Simple DGRL preprocessor using the proven working parse_dgrl_file function.
This extracts text line images from DGRL files for training.

Usage:
    python preprocess_dgrl.py --input_dir data/HWDB2.0Train --output_dir data/hwdb_processed
"""

import struct
import numpy as np
from PIL import Image
import os
import argparse
from multiprocessing import Pool, cpu_count


def parse_dgrl_file(filepath, debug=False):
    """
    Parse a single DGRL file and extract text lines with images.
    This is the PROVEN working parser from the git history.
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
            print(f"  {os.path.basename(filepath)}: {page_w}x{page_h}, {num_lines} lines")

        if num_lines > 50 or num_lines == 0:  # Sanity check
            return lines_data

        for line_idx in range(num_lines):
            if pos + 4 > len(data):
                break

            # Number of characters
            num_chars = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4

            if num_chars > 200 or num_chars == 0:  # Sanity check
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
            if height > 2000 or width > 8000 or height == 0 or width == 0:
                break

            # Read pixel data
            img_size = height * width
            if pos + img_size > len(data):
                break

            img_array = np.frombuffer(data[pos:pos+img_size], dtype=np.uint8)
            img_array = img_array.reshape(height, width)
            pos += img_size

            # Skip line trailing data until we hit 0xFF
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

            # For next line: search for valid header instead of relying on 0xFF
            # Look ahead up to 100KB for next num_chars (1-200 range)
            if line_idx < num_lines - 1:
                found_next = False
                search_limit = min(len(data) - 4, pos + 100000)

                for search_pos in range(pos, search_limit):
                    if search_pos + 4 > len(data):
                        break

                    candidate_chars = struct.unpack('<I', data[search_pos:search_pos+4])[0]

                    if 1 <= candidate_chars <= 200:
                        # Validate: check if followed by reasonable dimensions
                        validate_pos = search_pos + 4 + candidate_chars * 2
                        if validate_pos + 8 <= len(data):
                            val_h = struct.unpack('<I', data[validate_pos:validate_pos+4])[0]
                            val_w = struct.unpack('<I', data[validate_pos+4:validate_pos+8])[0]

                            if 1 <= val_h <= 2000 and 1 <= val_w <= 8000:
                                pos = search_pos
                                found_next = True
                                break

                if not found_next:
                    break  # No more valid lines found

    except Exception as e:
        if debug:
            print(f"  Error: {e}")

    return lines_data


def process_single_file(args):
    """Process one DGRL file (for multiprocessing)."""
    dgrl_path, output_dir, target_height = args
    base_name = os.path.splitext(os.path.basename(dgrl_path))[0]

    lines_data = parse_dgrl_file(dgrl_path)

    img_gt_pairs = []
    chars_set = set()

    for line_idx, line_data in enumerate(lines_data):
        # Resize to target height while maintaining aspect ratio
        img = line_data['image']
        h, w = img.shape
        ratio = target_height / h
        new_w = max(1, int(w * ratio))

        # Ensure minimum width for CTC (text_len * 20 pixels minimum)
        text_len = len(line_data['text'])
        min_width = text_len * 20
        if new_w < min_width:
            new_w = min_width

        # Resize using PIL
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)

        # Save image
        img_name = f"{base_name}_L{line_idx:03d}.png"
        img_path = os.path.join(output_dir, img_name)
        pil_img.save(img_path)

        # Collect metadata
        img_gt_pairs.append((img_name, line_data['text']))

        # Collect unique characters
        for char in line_data['text']:
            chars_set.add(char)

    return base_name, len(lines_data), img_gt_pairs, chars_set


def main():
    parser = argparse.ArgumentParser(description='Preprocess DGRL files to extract text line images')
    parser.add_argument('--input_dir', required=True, help='Input directory containing .dgrl files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed images')
    parser.add_argument('--target_height', type=int, default=128, help='Target image height (default: 128)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    # Find all DGRL files
    dgrl_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith('.dgrl')
    ]

    print(f"Found {len(dgrl_files)} DGRL files in {args.input_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process files in parallel
    process_args = [(f, args.output_dir, args.target_height) for f in dgrl_files]

    all_img_gt_pairs = []
    all_chars = set()
    total_lines = 0

    with Pool(min(args.workers, cpu_count())) as pool:
        results = pool.map(process_single_file, process_args)

    for base_name, line_count, img_gt_pairs, chars_set in results:
        all_img_gt_pairs.extend(img_gt_pairs)
        all_chars.update(chars_set)
        total_lines += line_count
        if line_count > 0:
            print(f"  ✓ {base_name}: {line_count} lines")

    # Save metadata
    metadata_path = os.path.join(args.output_dir, 'img_gt.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for img_name, text in all_img_gt_pairs:
            f.write(f"{img_name},{text}\n")

    # Save character list
    chars_path = os.path.join(args.output_dir, 'chars_list.txt')
    with open(chars_path, 'w', encoding='utf-8') as f:
        for char in sorted(all_chars):
            f.write(f"{char}\n")

    print(f"\n{'='*60}")
    print(f"✓ Processed {len(dgrl_files)} files")
    print(f"✓ Extracted {total_lines} text lines")
    print(f"✓ Unique characters: {len(all_chars)}")
    print(f"✓ Images saved to: {args.output_dir}")
    print(f"✓ Metadata: {metadata_path}")
    print(f"✓ Character list: {chars_path}")


if __name__ == '__main__':
    main()
