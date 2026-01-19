#!/usr/bin/env python3
"""
CASIA-HWDB .dgrl File Parser - Full Page Extraction

Converts .dgrl files (Offline Handwritten Page Data) to:
- PNG images of complete handwritten pages
- Text files with ground truth labels (one line per text line)
- Optional visualization with bounding boxes

This script extracts the FULL PAGE image and character-level annotations,
unlike preprocess_dgrl.py which extracts individual text line images.

Usage:
    python preprocess_dgrl_pages.py --input_dir data/HWDB2.0Train --output_dir data/pages
    python preprocess_dgrl_pages.py --input_dir data/HWDB2.0Train --output_dir data/pages --no-viz
"""

import os
import struct
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CharacterInfo:
    """Data structure for a single character's information"""
    def __init__(self, code: int, top: int, left: int, height: int, width: int):
        self.code = code
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def get_text(self) -> str:
        """Convert character code to text using GB18030 encoding"""
        try:
            return struct.pack('<H', self.code).decode('gb18030')
        except (UnicodeDecodeError, struct.error):
            return '�'  # Replacement character for invalid codes

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (left, top, right, bottom)"""
        return (self.left, self.top, self.left + self.width, self.top + self.height)


class LineInfo:
    """Data structure for a text line's information"""
    def __init__(self, top: int, left: int, height: int, width: int):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.characters: List[CharacterInfo] = []

    def add_character(self, char: CharacterInfo):
        """Add a character to this line"""
        self.characters.append(char)

    def get_text(self) -> str:
        """Get the complete text for this line"""
        return ''.join(char.get_text() for char in self.characters)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Return line bounding box as (left, top, right, bottom)"""
        return (self.left, self.top, self.left + self.width, self.top + self.height)


class DGRLPageParser:
    """
    Parser for .dgrl file format (full page extraction)

    File Structure:
    - Header Size (4 bytes, uint32)
    - Header Data (variable length)
    - Image Width (4 bytes, uint32)
    - Image Height (4 bytes, uint32)
    - Image Data (Width * Height bytes, grayscale bitmap)
    - Line Count (4 bytes, uint32)
    - For each line:
        - Character Count (4 bytes, uint32)
        - Line bbox: Top, Left, Height, Width (4 * int16 = 8 bytes)
        - For each character:
            - Char Code (2 bytes, uint16, GBK encoding)
            - Char bbox: Top, Left, Height, Width (4 * int16 = 8 bytes)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header_size = 0
        self.header_data = b''
        self.width = 0
        self.height = 0
        self.image_data = None
        self.lines: List[LineInfo] = []

    def parse(self) -> bool:
        """
        Parse the .dgrl file and extract all data

        Returns:
            bool: True if parsing succeeded, False otherwise
        """
        try:
            with open(self.filepath, 'rb') as f:
                # Read header size (4 bytes, little-endian unsigned int)
                header_size_bytes = f.read(4)
                if len(header_size_bytes) < 4:
                    print(f"✗ Error: File too short: {self.filepath}")
                    return False

                self.header_size = struct.unpack('<I', header_size_bytes)[0]

                # Read header data (variable length)
                self.header_data = f.read(self.header_size)
                if len(self.header_data) < self.header_size:
                    print(f"✗ Error: Incomplete header: {self.filepath}")
                    return False

                # Read image dimensions (2 * 4 bytes)
                width_bytes = f.read(4)
                height_bytes = f.read(4)

                if len(width_bytes) < 4 or len(height_bytes) < 4:
                    print(f"✗ Error: Missing dimensions: {self.filepath}")
                    return False

                self.width = struct.unpack('<I', width_bytes)[0]
                self.height = struct.unpack('<I', height_bytes)[0]

                # Validate dimensions (sanity check)
                if self.width <= 0 or self.height <= 0 or self.width > 10000 or self.height > 10000:
                    print(f"✗ Error: Invalid dimensions {self.width}x{self.height}: {self.filepath}")
                    return False

                # Read image data (width * height bytes)
                image_size = self.width * self.height
                image_bytes = f.read(image_size)

                if len(image_bytes) < image_size:
                    print(f"✗ Error: Incomplete image (expected {image_size}, got {len(image_bytes)}): {self.filepath}")
                    return False

                # Convert to numpy array
                self.image_data = np.frombuffer(image_bytes, dtype=np.uint8).reshape(self.height, self.width)

                # Read line count (4 bytes)
                line_count_bytes = f.read(4)
                if len(line_count_bytes) < 4:
                    print(f"✗ Warning: Missing line count (assuming 0 lines): {self.filepath}")
                    return True  # Still valid, just no annotations

                line_count = struct.unpack('<I', line_count_bytes)[0]

                # Sanity check on line count
                if line_count > 1000:
                    print(f"✗ Warning: Suspiciously high line count ({line_count}), truncating: {self.filepath}")
                    line_count = 1000

                # Read each line
                for line_idx in range(line_count):
                    # Read character count (4 bytes)
                    char_count_bytes = f.read(4)
                    if len(char_count_bytes) < 4:
                        print(f"⚠ Warning: Incomplete line {line_idx}: {self.filepath}")
                        break

                    char_count = struct.unpack('<I', char_count_bytes)[0]

                    # Sanity check on character count
                    if char_count > 500:
                        print(f"⚠ Warning: Suspiciously high char count ({char_count}) at line {line_idx}: {self.filepath}")
                        break

                    # Read line bounding box (4 shorts = 8 bytes)
                    line_bbox_bytes = f.read(8)
                    if len(line_bbox_bytes) < 8:
                        print(f"⚠ Warning: Incomplete line bbox at line {line_idx}: {self.filepath}")
                        break

                    line_top, line_left, line_height, line_width = struct.unpack('<4h', line_bbox_bytes)
                    line_info = LineInfo(line_top, line_left, line_height, line_width)

                    # Read each character in this line
                    for char_idx in range(char_count):
                        # Read character: code (2 bytes) + bbox (8 bytes) = 10 bytes
                        char_data = f.read(10)
                        if len(char_data) < 10:
                            print(f"⚠ Warning: Incomplete char at line {line_idx}, char {char_idx}: {self.filepath}")
                            break

                        # Unpack character code (uint16)
                        char_code = struct.unpack('<H', char_data[0:2])[0]

                        # Unpack character bbox (4 * int16)
                        char_top, char_left, char_height, char_width = struct.unpack('<4h', char_data[2:10])

                        char_info = CharacterInfo(char_code, char_top, char_left, char_height, char_width)
                        line_info.add_character(char_info)

                    self.lines.append(line_info)

                return True

        except Exception as e:
            print(f"✗ Error parsing {self.filepath}: {e}")
            return False

    def save_image(self, output_path: str):
        """Save the full page image as PNG"""
        if self.image_data is None:
            raise ValueError("No image data. Parse file first.")

        img = Image.fromarray(self.image_data, mode='L')
        img.save(output_path, 'PNG')

    def save_labels(self, output_path: str):
        """Save text labels to file (one line per text line)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in self.lines:
                f.write(line.get_text() + '\n')

    def save_visualization(self, output_path: str):
        """Save visualization with bounding boxes"""
        if self.image_data is None:
            raise ValueError("No image data. Parse file first.")

        # Convert grayscale to RGB for colored boxes
        img = Image.fromarray(self.image_data, mode='L').convert('RGB')
        draw = ImageDraw.Draw(img)

        # Try to load Chinese font, fall back to default
        try:
            # macOS Chinese font
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 16)
        except:
            try:
                # Linux alternative
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 16)
            except:
                font = ImageFont.load_default()

        # Draw line bounding boxes (blue, thicker)
        for line in self.lines:
            line_bbox = line.get_bbox()
            draw.rectangle(line_bbox, outline='blue', width=2)

            # Draw character bounding boxes (red, thinner)
            for char in line.characters:
                char_bbox = char.get_bbox()
                draw.rectangle(char_bbox, outline='red', width=1)

        img.save(output_path, 'PNG')

    def get_summary(self) -> dict:
        """Get summary statistics"""
        return {
            'width': self.width,
            'height': self.height,
            'line_count': len(self.lines),
            'char_count': sum(len(line.characters) for line in self.lines),
            'text': '\n'.join(line.get_text() for line in self.lines)
        }


def process_dgrl_directory(
    input_dir: str,
    output_dir: str,
    create_viz: bool = True,
    verbose: bool = False
):
    """
    Process all .dgrl files in a directory

    Args:
        input_dir: Directory containing .dgrl files
        output_dir: Directory for output files
        create_viz: Whether to create visualization images
        verbose: Print detailed progress
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory structure
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if create_viz:
        viz_dir = output_path / 'viz'
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Find all .dgrl files
    dgrl_files = sorted(input_path.glob('*.dgrl'))

    if not dgrl_files:
        print(f"✗ No .dgrl files found in {input_dir}")
        return

    print(f"Found {len(dgrl_files)} .dgrl files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization: {'enabled' if create_viz else 'disabled'}")
    print("=" * 60)

    # Process each file
    success_count = 0
    fail_count = 0
    total_lines = 0
    total_chars = 0

    for idx, dgrl_file in enumerate(dgrl_files, 1):
        base_name = dgrl_file.stem

        if verbose or idx % 10 == 0:
            print(f"[{idx}/{len(dgrl_files)}] Processing {dgrl_file.name}...", end=' ')

        try:
            parser = DGRLPageParser(str(dgrl_file))

            if not parser.parse():
                if verbose:
                    print("FAILED")
                fail_count += 1
                continue

            # Save outputs
            parser.save_image(str(images_dir / f"{base_name}.png"))
            parser.save_labels(str(labels_dir / f"{base_name}.txt"))

            if create_viz:
                parser.save_visualization(str(viz_dir / f"{base_name}_viz.png"))

            # Collect statistics
            summary = parser.get_summary()
            total_lines += summary['line_count']
            total_chars += summary['char_count']

            if verbose or idx % 10 == 0:
                print(f"✓ ({summary['width']}x{summary['height']}, {summary['line_count']} lines)")

            success_count += 1

        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")
            fail_count += 1

    # Final summary
    print("=" * 60)
    print(f"✓ Processing complete:")
    print(f"  Success:        {success_count}")
    print(f"  Failed:         {fail_count}")
    print(f"  Total lines:    {total_lines}")
    print(f"  Total chars:    {total_chars}")
    print(f"\n✓ Output saved to: {output_path}")
    print(f"  - Images:       {images_dir}")
    print(f"  - Labels:       {labels_dir}")
    if create_viz:
        print(f"  - Visualization: {viz_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CASIA-HWDB .dgrl Full Page Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract pages with visualization
  python preprocess_dgrl_pages.py --input_dir data/HWDB2.0Train --output_dir data/pages

  # Extract without visualization (faster)
  python preprocess_dgrl_pages.py --input_dir data/HWDB2.0Train --output_dir data/pages --no-viz

  # Verbose output
  python preprocess_dgrl_pages.py --input_dir data/HWDB2.0Train --output_dir data/pages --verbose
        """
    )

    parser.add_argument(
        '--input_dir',
        required=True,
        help='Input directory containing .dgrl files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization generation (faster processing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress for each file'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"✗ Error: Input directory does not exist: {args.input_dir}")
        return 1

    # Process files
    process_dgrl_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        create_viz=not args.no_viz,
        verbose=args.verbose
    )

    return 0


if __name__ == '__main__':
    exit(main())
