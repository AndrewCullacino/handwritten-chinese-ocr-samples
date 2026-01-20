#!/usr/bin/env python3
'''
Extract text line images from DGRL files (CASIA-HWDB2.x format)
Based on official DGRL format specification from:
    http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
    DGRLRead.cpp.pdf

DGRL Format Structure:
    File Header (36 + len(illustr) bytes):
        - Header size (4B): 36 + strlen(illustr)
        - Format code (8B): "DGRL"
        - Illustration (variable): "#......\0"
        - Code type (20B): "ASCII", "GB", etc.
        - Code length (2B): 1, 2, 4, etc.
        - Bits per pixel (2B): 1 (B/W) or 8 (Gray)
    
    Image Records:
        - Image height (4B)
        - Image width (4B)
        - Line number (4B)
    
    Line Records (for each line):
        - Char number (4B)
        - Labels (code_length * char_number bytes)
        - Top position (4B)
        - Left position (4B)
        - Height (4B)
        - Width (4B)
        - Bitmap (H * W bytes for gray, H * ((W+7)/8) for binary)

Usage:
    python dgrl2png.py <source_file_or_dir> <target_folder> [--image_height 64]
    
    # Process a single DGRL file
    python dgrl2png.py /path/to/001-P001.dgrl ./extracted_data
    
    # Process a directory of DGRL files
    python dgrl2png.py /path/to/HWDB2.0Train ./extracted_data
    
    # Process a zip file containing DGRL files
    python dgrl2png.py /path/to/HWDB2.0Train.zip ./extracted_data
'''

import numpy as np
import os
import struct
import sys
import zipfile
import codecs
import argparse
from typing import Tuple, List, Optional
from PIL import Image


def read_dgrl_header(fp) -> Tuple[int, str, str, int, int]:
    """
    Read DGRL file header.
    
    Returns:
        header_size: Size of header in bytes
        format_code: Should be "DGRL"
        code_type: Character encoding type ("GB", "ASCII", etc.)
        code_length: Bytes per character code (typically 2 for GB)
        bits_per_pixel: 1 for binary, 8 for grayscale
    """
    # Read header size (4 bytes)
    header_size = struct.unpack('<I', fp.read(4))[0]
    
    # Read format code (8 bytes)
    format_code = fp.read(8).decode('ascii', errors='ignore').rstrip('\x00')
    
    # Read illustration (variable length: header_size - 36)
    illust_len = header_size - 36
    illustration = fp.read(illust_len).decode('ascii', errors='ignore').rstrip('\x00')
    
    # Read code type (20 bytes)
    code_type = fp.read(20).decode('ascii', errors='ignore').rstrip('\x00')
    
    # Read code length (2 bytes)
    code_length = struct.unpack('<H', fp.read(2))[0]
    
    # Read bits per pixel (2 bytes)
    bits_per_pixel = struct.unpack('<H', fp.read(2))[0]
    
    return header_size, format_code, code_type, code_length, bits_per_pixel


def decode_label(label_bytes: bytes, code_type: str) -> str:
    """
    Decode label bytes to string based on code type.
    
    Args:
        label_bytes: Raw bytes of character codes
        code_type: Encoding type ("GB", "GBK", "ASCII", etc.)
    
    Returns:
        Decoded string
    """
    # Replace garbage bytes (0xFF) with empty
    label_bytes = bytes(b if b != 0xff else 0x20 for b in label_bytes)
    
    text = ''
    i = 0
    while i < len(label_bytes):
        # Check for single-byte character (ASCII or padding)
        if label_bytes[i] < 0x80:
            if label_bytes[i] != 0x00 and label_bytes[i] != 0x20:
                text += chr(label_bytes[i])
            i += 1
        else:
            # Double-byte character (GB/GBK)
            if i + 1 < len(label_bytes):
                try:
                    char_bytes = bytes([label_bytes[i], label_bytes[i + 1]])
                    char = char_bytes.decode('gb18030', errors='replace')
                    if char != '\ufffd':
                        text += char
                except:
                    pass
                i += 2
            else:
                i += 1
    
    return text


def read_dgrl_page(fp, code_length: int, code_type: str, bits_per_pixel: int) -> List[dict]:
    """
    Read a page of DGRL file and extract all text lines.
    
    Args:
        fp: File pointer positioned after header
        code_length: Bytes per character code
        code_type: Character encoding type
        bits_per_pixel: 1 for binary, 8 for grayscale
    
    Returns:
        List of dicts with 'image', 'text', 'top', 'left', 'height', 'width'
    """
    lines_data = []
    
    # Read page dimensions
    page_height = struct.unpack('<I', fp.read(4))[0]
    page_width = struct.unpack('<I', fp.read(4))[0]
    line_number = struct.unpack('<I', fp.read(4))[0]
    
    # Sanity check
    if line_number > 100 or line_number == 0:
        return lines_data
    
    for line_idx in range(line_number):
        try:
            # Read number of characters in this line
            char_number = struct.unpack('<I', fp.read(4))[0]
            
            if char_number > 500 or char_number == 0:
                break
            
            # Read character labels
            label_size = char_number * code_length
            label_bytes = fp.read(label_size)
            text = decode_label(label_bytes, code_type)
            
            # Read line position (top, left)
            line_top = struct.unpack('<I', fp.read(4))[0]
            line_left = struct.unpack('<I', fp.read(4))[0]
            
            # Read line dimensions (height, width)
            line_height = struct.unpack('<I', fp.read(4))[0]
            line_width = struct.unpack('<I', fp.read(4))[0]
            
            # Sanity check dimensions
            if line_height > 5000 or line_width > 10000 or line_height == 0 or line_width == 0:
                break
            
            # Read bitmap data
            if bits_per_pixel == 1:
                # Binary image: H * ((W + 7) / 8) bytes
                row_bytes = (line_width + 7) // 8
                bitmap_size = line_height * row_bytes
                bitmap_data = fp.read(bitmap_size)
                
                # Convert binary to grayscale
                img_array = np.zeros((line_height, line_width), dtype=np.uint8)
                for row in range(line_height):
                    for col in range(line_width):
                        byte_idx = row * row_bytes + col // 8
                        bit_idx = 7 - (col % 8)
                        if byte_idx < len(bitmap_data):
                            pixel = (bitmap_data[byte_idx] >> bit_idx) & 1
                            img_array[row, col] = 255 if pixel == 0 else 0
            else:
                # Grayscale image: H * W bytes
                bitmap_size = line_height * line_width
                bitmap_data = fp.read(bitmap_size)
                img_array = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(line_height, line_width)
            
            if text:  # Only add if we have valid text
                lines_data.append({
                    'image': img_array,
                    'text': text,
                    'top': line_top,
                    'left': line_left,
                    'height': line_height,
                    'width': line_width
                })
                
        except Exception as e:
            print(f"  Warning: Error reading line {line_idx}: {e}")
            break
    
    return lines_data


def process_dgrl_file(fp, file_name: str, tgt_folder: str, 
                      target_height: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Process a single DGRL file and extract text line images.
    
    Args:
        fp: File pointer to DGRL file
        file_name: Name of the DGRL file (for naming output files)
        tgt_folder: Target folder for extracted images
        target_height: Optional target height for resizing images
    
    Returns:
        Tuple of (list of output image paths, list of labels in hex codes)
    """
    output_paths = []
    output_labels = []
    
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    try:
        # Read header
        header_size, format_code, code_type, code_length, bits_per_pixel = read_dgrl_header(fp)
        
        if format_code.upper() not in ['DGRL', 'DGR']:
            print(f"  Warning: Unexpected format code '{format_code}' in {file_name}")
        
        # Read all lines from the page
        lines_data = read_dgrl_page(fp, code_length, code_type, bits_per_pixel)
        
        for line_idx, line_data in enumerate(lines_data):
            img = line_data['image']
            text = line_data['text']
            
            # Resize if target height specified
            if target_height is not None and img.shape[0] > 0:
                h, w = img.shape
                ratio = target_height / h
                new_w = max(1, int(w * ratio))
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
                img = np.array(pil_img)
            
            # Generate output filename
            img_filename = f"{base_name}-L{line_idx + 1}.png"
            img_path = os.path.join(tgt_folder, img_filename)
            
            # Save image
            pil_img = Image.fromarray(img)
            pil_img.save(img_path)
            
            # Save label file (hex codes for each character)
            label_filename = f"{base_name}-L{line_idx + 1}.txt"
            label_path = os.path.join(tgt_folder, label_filename)
            with open(label_path, 'w', encoding='utf-8') as label_file:
                for char in text:
                    try:
                        # Convert character to hex code
                        char_bytes = char.encode('gb18030')
                        hex_code = char_bytes.hex().upper()
                        label_file.write(hex_code + '\n')
                    except:
                        pass
            
            output_paths.append(img_path)
            output_labels.append(text)
            
    except Exception as e:
        print(f"  Error processing {file_name}: {e}")
    
    return output_paths, output_labels


def dgrl2png(src_path: str, tgt_folder: str, target_height: Optional[int] = None):
    """
    Main function to extract images from DGRL files.
    
    Args:
        src_path: Path to DGRL file, directory of DGRL files, or zip archive
        tgt_folder: Target folder for extracted images
        target_height: Optional target height for resizing images
    """
    os.makedirs(tgt_folder, exist_ok=True)
    
    all_paths = []
    all_labels = []
    
    if zipfile.is_zipfile(src_path):
        # Process zip file
        print(f"Processing zip file: {src_path}")
        with zipfile.ZipFile(src_path, 'r') as zip_file:
            file_list = [f for f in zip_file.namelist() if f.lower().endswith('.dgrl')]
            print(f"Found {len(file_list)} DGRL files in archive")
            
            for file_name in file_list:
                print(f"  Processing {file_name}...")
                with zip_file.open(file_name) as data_file:
                    paths, labels = process_dgrl_file(data_file, file_name, tgt_folder, target_height)
                    all_paths.extend(paths)
                    all_labels.extend(labels)
                    if paths:
                        print(f"    Extracted {len(paths)} lines")
                        
    elif os.path.isdir(src_path):
        # Process directory
        dgrl_files = [f for f in os.listdir(src_path) if f.lower().endswith('.dgrl')]
        print(f"Found {len(dgrl_files)} DGRL files in directory")
        
        for file_name in dgrl_files:
            file_path = os.path.join(src_path, file_name)
            print(f"  Processing {file_name}...")
            with open(file_path, 'rb') as fp:
                paths, labels = process_dgrl_file(fp, file_name, tgt_folder, target_height)
                all_paths.extend(paths)
                all_labels.extend(labels)
                if paths:
                    print(f"    Extracted {len(paths)} lines")
                    
    elif os.path.isfile(src_path):
        # Process single file
        print(f"Processing file: {src_path}")
        with open(src_path, 'rb') as fp:
            paths, labels = process_dgrl_file(fp, src_path, tgt_folder, target_height)
            all_paths.extend(paths)
            all_labels.extend(labels)
            if paths:
                print(f"  Extracted {len(paths)} lines")
    else:
        print(f"Error: {src_path} is not a valid file or directory")
        return
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Total images extracted: {len(all_paths)}")
    print(f"  Output directory: {tgt_folder}")
    
    # Generate image-groundtruth file
    img_gt_path = os.path.join(tgt_folder, 'dgrl_img_gt.txt')
    with open(img_gt_path, 'w', encoding='utf-8') as f:
        for path, label in zip(all_paths, all_labels):
            img_name = os.path.basename(path)
            f.write(f"{img_name},{label}\n")
    print(f"  Image-GT file: {img_gt_path}")


def generate_dgrl_file_list(src_path: str, output_file: str):
    """
    Generate a list of DGRL files (similar to hwdb2x_train_dgrs.txt).
    
    Args:
        src_path: Path to directory or zip file containing DGRL files
        output_file: Output file path for the list
    """
    dgrl_files = []
    
    if zipfile.is_zipfile(src_path):
        with zipfile.ZipFile(src_path, 'r') as zip_file:
            dgrl_files = [f for f in zip_file.namelist() if f.lower().endswith('.dgrl')]
    elif os.path.isdir(src_path):
        dgrl_files = [os.path.join(src_path, f) for f in os.listdir(src_path) 
                      if f.lower().endswith('.dgrl')]
    
    with open(output_file, 'w') as f:
        for dgrl_file in sorted(dgrl_files):
            f.write(dgrl_file + '\n')
    
    print(f"Generated file list with {len(dgrl_files)} entries: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract text line images from DGRL files (CASIA-HWDB2.x format)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Process a single DGRL file
    python dgrl2png.py /path/to/001-P001.dgrl ./extracted_data
    
    # Process a directory of DGRL files
    python dgrl2png.py /path/to/HWDB2.0Train ./extracted_data
    
    # Process a zip file with custom image height
    python dgrl2png.py HWDB2.0Train.zip ./extracted_data --image_height 64
    
    # Generate file list
    python dgrl2png.py /path/to/HWDB2.0Train --list_only hwdb2x_dgrl_files.txt
        '''
    )
    
    parser.add_argument('source', help='Source DGRL file, directory, or zip archive')
    parser.add_argument('target', nargs='?', default='./extracted_dgrl_data',
                        help='Target folder for extracted images (default: ./extracted_dgrl_data)')
    parser.add_argument('--image_height', type=int, default=None,
                        help='Target image height (maintains aspect ratio)')
    parser.add_argument('--list_only', type=str, default=None,
                        help='Only generate file list, do not extract images')
    
    args = parser.parse_args()
    
    if args.list_only:
        generate_dgrl_file_list(args.source, args.list_only)
    else:
        dgrl2png(args.source, args.target, args.image_height)
