#!/usr/bin/env python3
"""
Debug script to analyze DGRL file structure.
Run this in Colab to understand why only 1 line is extracted per file.
"""

import os
import struct

def debug_dgrl_file(filepath, max_lines=3):
    """Analyze a DGRL file structure in detail."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data)} bytes")
    
    # Show first 100 bytes in hex
    print(f"\nFirst 100 bytes (hex):")
    for i in range(0, min(100, len(data)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"  {i:04x}: {hex_str:<48} {ascii_str}")
    
    # Parse header
    pos = 73
    page_w = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    page_h = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    num_lines = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    
    print(f"\nHeader info:")
    print(f"  Position 73-84: page_w={page_w}, page_h={page_h}, num_lines={num_lines}")
    
    if num_lines > 50:
        print(f"  WARNING: num_lines={num_lines} seems wrong (>50)")
        return
    
    print(f"\nParsing {min(max_lines, num_lines)} lines:")
    
    for line_idx in range(min(max_lines, num_lines)):
        print(f"\n  --- Line {line_idx} at pos {pos} ---")
        
        if pos + 4 > len(data):
            print(f"  ERROR: Not enough data for num_chars")
            break
        
        num_chars = struct.unpack('<I', data[pos:pos+4])[0]
        print(f"  num_chars: {num_chars}")
        pos += 4
        
        if num_chars > 200:
            print(f"  ERROR: num_chars={num_chars} > 200, breaking")
            # Show next 20 bytes for debugging
            print(f"  Next 20 bytes: {data[pos-4:pos+16].hex()}")
            break
        
        if num_chars == 0:
            print(f"  ERROR: num_chars=0, breaking")
            break
        
        # Read text
        text = ''
        for i in range(num_chars):
            code = data[pos:pos+2]
            pos += 2
            try:
                char = code.decode('gb18030')
                text += char
            except:
                text += '?'
        print(f"  text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        if pos + 8 > len(data):
            print(f"  ERROR: Not enough data for dimensions")
            break
        
        height = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        width = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        print(f"  dimensions: {width}x{height}")
        
        if height > 2000 or width > 8000 or height == 0 or width == 0:
            print(f"  ERROR: Invalid dimensions, breaking")
            break
        
        img_size = height * width
        print(f"  image_size: {img_size} bytes")
        
        if pos + img_size > len(data):
            print(f"  ERROR: Not enough data for image")
            break
        
        pos += img_size
        print(f"  After image, pos={pos}")
        
        # Show next 20 bytes
        next_bytes = data[pos:pos+20]
        print(f"  Next 20 bytes: {next_bytes.hex()}")
        
        # Skip trailing data until 0xFF
        trailing_count = 0
        while pos < len(data) and data[pos] != 0xFF:
            trailing_count += 1
            pos += 1
        print(f"  Skipped {trailing_count} trailing bytes")
        
        # Skip 0xFF padding
        ff_count = 0
        while pos < len(data) and data[pos] == 0xFF:
            ff_count += 1
            pos += 1
        print(f"  Skipped {ff_count} 0xFF padding bytes")
        
        print(f"  Ready for next line at pos={pos}")


# Main - run in Colab
if __name__ == '__main__':
    import sys
    
    # Default path for Colab
    drive_root = '/content/drive/MyDrive/handwritten-chinese-ocr-samples'
    
    # Find first DGRL file
    train_folders = [
        os.path.join(drive_root, 'HWDB2.0Train'),
        os.path.join(drive_root, 'HWDB2.1Train'),
    ]
    
    for folder in train_folders:
        if os.path.exists(folder):
            dgrl_files = [f for f in os.listdir(folder) if f.endswith('.dgrl')]
            if dgrl_files:
                # Analyze first file
                debug_dgrl_file(os.path.join(folder, dgrl_files[0]))
                break
