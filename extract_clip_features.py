#!/usr/bin/env python3
"""
Extract image visual features using OpenAI CLIP and save per-image .npy files.

Usage:
  python scripts/extract_clip_features.py /path/to/images /path/to/features --model ViT-B/32 --batch-size 32

Dependencies:
  pip install torch torchvision pillow numpy tqdm
  pip install git+https://github.com/openai/CLIP.git

The script finds all PNG files under the source directory (non-recursive by default),
extracts CLIP image features and saves each feature as `basename.npy` in the
destination directory (preserving the base filename).
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

try:
    import clip
except Exception as e:
    raise ImportError(
        "Cannot import `clip`. Install with: pip install git+https://github.com/openai/CLIP.git"
    )


def load_image_rgb(path: Path):
    img = Image.open(path).convert("RGB")
    return img


def run(src_dir, dst_dir, model_name='ViT-B/16', batch_size=32, device=None, recursive=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # Collect files
    if recursive:
        files = sorted([p for p in src.rglob('*.png') if p.is_file()])
    else:
        files = sorted([p for p in src.glob('*.png') if p.is_file()])

    if len(files) == 0:
        print(f"No PNG files found in {src}")
        return

    print(f"Found {len(files)} images. Extracting features with {model_name} on {device}...")

    with torch.no_grad():
        for i in tqdm(range(0, len(files), batch_size), desc="batches"):
            batch_files = files[i:i+batch_size]
            imgs = []
            for p in batch_files:
                img = load_image_rgb(p)
                imgs.append(preprocess(img).unsqueeze(0))
            batch_tensor = torch.cat(imgs, dim=0).to(device)

            # encode
            feats = model.encode_image(batch_tensor)
            # move to cpu numpy
            feats = feats.cpu().numpy()

            # save per-image
            for j, p in enumerate(batch_files):
                out_name = dst / (p.stem + '.npy')
                np.save(out_name, feats[j])

    print(f"Saved features to {dst}")

     


def parse_args():
    ap = argparse.ArgumentParser(description="Extract CLIP features for PNG images")
    ap.add_argument('--src', default="../dataset/NUAA-SIRST/images", help='Source directory containing PNG images')
    ap.add_argument('--dst',default="../dataset/NUAA-SIRST/features/L14", help='Destination directory to save features (.npy)')
    ap.add_argument('--model', default='ViT-L/14', help='CLIP model to use (default: ViT-B/32)')
    ap.add_argument('--batch-size', type=int, default=32, help='Batch size for feature extraction')
    ap.add_argument('--device', default="cuda:1", help='Torch device (cpu or cuda). Auto-detect if omitted')
    ap.add_argument('--recursive', action='store_true', help=' Search PNGs recursively under src')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.src, args.dst, model_name=args.model, batch_size=args.batch_size, device=args.device, recursive=args.recursive)
