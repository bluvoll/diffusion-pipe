#!/usr/bin/env python3
"""Generate dataset.toml for diffusion-pipe training.

Auto-discovers subdirectories containing images and generates a complete
dataset.toml with proper header settings and [[directory]] entries.

Usage:
    python3 gen_dataset.py /path/to/data > dataset.toml
    python3 gen_dataset.py /path/to/data --repeats 2 --resolution 512
    python3 gen_dataset.py /path/to/data -r 768 -n 3 -o dataset.toml
"""

import argparse
import os
import sys

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def find_image_dirs(root):
    """Find all directories under root that contain at least one image file."""
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                dirs.append(dirpath)
                break
    dirs.sort()
    return dirs


def generate_toml(root, resolution, num_repeats, min_ar, max_ar, num_ar_buckets, enable_ar_bucket):
    lines = []
    lines.append(f'resolutions = [{resolution}]')
    lines.append(f'enable_ar_bucket = {"true" if enable_ar_bucket else "false"}')
    lines.append(f'min_ar = {min_ar}')
    lines.append(f'max_ar = {max_ar}')
    lines.append(f'num_ar_buckets = {num_ar_buckets}')

    dirs = find_image_dirs(root)
    if not dirs:
        print(f'Warning: No directories with images found under {root}', file=sys.stderr)
        return ''

    print(f'Found {len(dirs)} directories with images', file=sys.stderr)

    for d in dirs:
        lines.append('')
        lines.append('[[directory]]')
        lines.append(f"path = '{d}/'")
        lines.append(f'num_repeats = {num_repeats}')

    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset.toml for diffusion-pipe')
    parser.add_argument('root', help='Root directory containing image subdirectories')
    parser.add_argument('-r', '--resolution', type=int, default=512, help='Training resolution (default: 512)')
    parser.add_argument('-n', '--repeats', type=int, default=1, help='Number of repeats per directory (default: 1)')
    parser.add_argument('--min-ar', type=float, default=0.5, help='Minimum aspect ratio (default: 0.5)')
    parser.add_argument('--max-ar', type=float, default=2.0, help='Maximum aspect ratio (default: 2.0)')
    parser.add_argument('--num-ar-buckets', type=int, default=7, help='Number of AR buckets (default: 7)')
    parser.add_argument('--no-ar-bucket', action='store_true', help='Disable aspect ratio bucketing')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f'Error: {root} is not a directory', file=sys.stderr)
        sys.exit(1)

    toml = generate_toml(
        root,
        resolution=args.resolution,
        num_repeats=args.repeats,
        min_ar=args.min_ar,
        max_ar=args.max_ar,
        num_ar_buckets=args.num_ar_buckets,
        enable_ar_bucket=not args.no_ar_bucket,
    )

    if args.output:
        with open(args.output, 'w') as f:
            f.write(toml)
        print(f'Written to {args.output}', file=sys.stderr)
    else:
        print(toml)


if __name__ == '__main__':
    main()
