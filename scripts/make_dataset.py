#!/usr/bin/env python3
"""
For practical reasons and specifically because the dataset creation process was previously implemented in the Mask R-CNN project before, this process needs to be performed in that project.

make_dataset.py

Script to create a processed dataset of tooth images
from raw oral cavity images, using annotations for
bounding boxes and plaque/no-plaque labels.
"""

import os
import argparse
import json

# Import your data pipeline functions
from src.data_pipeline import (
    load_annotations,
    preprocess_and_label,
    split_dataset
)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a processed dataset of tooth images."
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Path to the CSV (or JSON) file with bounding boxes and labels."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where processed images will be saved."
    )
    parser.add_argument(
        "--mask-bg",
        action="store_true",
        help="If set, mask the background in cropped images."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="If set, also split the resulting dataset into train/val/test."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data to use for training if splitting."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation if splitting."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data to use for testing if splitting."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 1. Load annotations
    print(f"[INFO] Loading annotations from: {args.annotation_file}")
    df_annotations = load_annotations(args.annotation_file)

    # 2. Preprocess images (crop, mask if requested) and label
    print("[INFO] Preprocessing and labeling images...")
    preprocess_and_label(
        df_annotations=df_annotations,
        output_dir=args.output_dir,
        mask_bg=args.mask_bg
    )
    print("[INFO] Preprocessing complete.")

    # 3. Optionally split the dataset
    if args.split:
        print("[INFO] Splitting the dataset into train/val/test...")
        splits = split_dataset(
            images_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=42
        )
        # You might want to save split information to a JSON or CSV for reference
        split_info_path = os.path.join(args.output_dir, "dataset_splits.json")
        with open(split_info_path, "w") as f:
            # Convert lists of tuples into a JSON-friendly structure
            # e.g. { "train": [ {"path": ..., "label": ...}, ... ], ... }
            json_splits = {}
            for split_name, sample_list in splits.items():
                json_splits[split_name] = [
                    {"path": p, "label": l} for (p, l) in sample_list
                ]
            json.dump(json_splits, f, indent=2)
        
        print(f"[INFO] Split details saved to {split_info_path}")

    print("[INFO] Dataset generation complete.")


if __name__ == "__main__":
    main()
