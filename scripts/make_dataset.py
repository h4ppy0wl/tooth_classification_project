#!/usr/bin/env python3
"""
For practical reasons and specifically because the dataset creation process 
was previously implemented in the Mask R-CNN project before, this process needs
to be performed in that project.

make_dataset.py

Script to create a processed dataset of tooth images
from raw oral cavity images, using annotations for
bounding boxes and plaque/no-plaque labels.
"""

import os
import sys
import argparse
import json
from dataclasses import dataclass, asdict

# Import your data pipeline functions
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import src.data_pipeline as dp
from src.config import Config


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a processed dataset of tooth images and annotations."
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="If set, filters the raw dataset and stores the filtered and converted json alongside the filtered images list."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="If set, also split the resulting dataset into train/val/test and stores respective jsons."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    my_config = Config()
    if args.filter:
        # 1. Load annotations
        raw_path = os.path.join(my_config.DATA_DIR, my_config.RAW_DS_DIR)
        intrim_path = os.path.join(my_config.DATA_DIR, my_config.INTRIM_DIR)
        print(f"[INFO] Loading raw dataset from: {raw_path}")
        dp.preprocess_raw_dataset(raw_path, intrim_path, my_config , verbose= True)

    # 2. Optionally split the dataset
    if args.split:
        print("[INFO] Splitting the dataset into train/val/test...")
        master_json_path = os.path.join(intrim_path, f"filtered_{my_config.TARGET_CLASS}_annotations.json")
        processed_path = os.path.join(my_config.DATA_DIR, my_config.OUTPUT_DIR)
        dp.split_dataset_json(master_json_path, processed_path, my_config, verbose = True)
        # You might want to save split information to a JSON or CSV for reference
        split_info_path = os.path.join(processed_path, "dataset_splits_info.text")
        with open(split_info_path, "w", encoding = 'utf-8') as f:
            f.write(f"source: {master_json_path}\n")
            for field, value in asdict(my_config).items():
                f.write(f"{field}: {value}\n")
        
        print(f"[INFO] All configuration including split and stratifying details saved to {split_info_path}")

    print("[INFO] Dataset generation complete.")


if __name__ == "__main__":
    main()
