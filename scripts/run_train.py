import os
import sys
import json
import time
import tensorflow as tf

# Add parent and current directories to sys.path if needed.
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)

from src.config import Config
from src.data_pipeline import (
    parse_dataset_json,
    preprocess_and_save_images,
    build_tf_dataset_from_preprocessed
)
from src.model import create_model        
from src.train import train_transfer_model

def main():
    overall_start = time.time()
    run_num = 1
    # 1. Load configuration.
    t0 = time.time()
    config = Config()
    print(f"[Step 1] Loaded configuration in {time.time() - t0:.2f} seconds.")
    
    # Construct paths to the annotation JSON files.
    t0 = time.time()
    train_json = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.TRAIN_JSON_NAME)
    val_json   = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.VAL_JSON_NAME)
    print(f"[Step 2] Constructed JSON paths in {time.time() - t0:.2f} seconds.")
    
    # 2. Parse records from JSON files.
    t0 = time.time()
    print("Parsing training and validation JSON files...")
    train_records = parse_dataset_json(train_json, config, is_train_ds=True)
    val_records   = parse_dataset_json(val_json, config, is_train_ds=False)
    print(f"[Step 3] Parsed JSON records in {time.time() - t0:.2f} seconds.")
    
    # 3. Preprocess images and save them (or load preprocessed records if already done).
    t0 = time.time()
    print("Checking for preprocessed data and processing if necessary...")
    train_records_updated, _ = preprocess_and_save_images(train_records, config, "train")
    val_records_updated, _   = preprocess_and_save_images(val_records, config, "val")
    print(f"[Step 4] Preprocessed and saved images (or loaded saved records) in {time.time() - t0:.2f} seconds.")
    
    # 4. Build TensorFlow datasets from the updated (preprocessed) records.
    t0 = time.time()
    print("Building TensorFlow datasets from preprocessed records...")
    train_ds = build_tf_dataset_from_preprocessed(train_records_updated, config)
    val_ds   = build_tf_dataset_from_preprocessed(val_records_updated, config)
    print(f"[Step 5] Built TF datasets in {time.time() - t0:.2f} seconds.")
    
    # 5. Check GPU availability.
    t0 = time.time()
    gpus = tf.config.list_physical_devices('GPU')
    print("Physical devices available (GPU):", gpus)
    print(f"[Step 6] Checked GPU availability in {time.time() - t0:.2f} seconds.")
    
    # 6. Create the transfer learning model.
    t0 = time.time()
    print("Creating the model...")
    my_model = create_model(config, 'transfer', trainable_base=False)
    print(f"[Step 7] Created the model in {time.time() - t0:.2f} seconds.")
    
    # 7. Train the model.
    t0 = time.time()
    print("Starting training...")
    trained_model, h1, h2, current_log_dir = train_transfer_model(my_model, train_ds, val_ds, config, save_models=True, num = run_num)
    print(f"[Step 8] Model training completed in {time.time() - t0:.2f} seconds.")
    
    overall_time = time.time() - overall_start
    print(f"Overall training process took {overall_time:.2f} seconds.")

    # ---------------------------
    # Testing Phase
    # ---------------------------
    
    t0 = time.time()
    print("Starting test evaluation...")
    
    # Construct the test JSON file path.
    test_json = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.TEST_JSON_NAME)
    print(f"[Test Step] Constructed test JSON path in {time.time() - t0:.2f} seconds.")
    
    # Parse test records from the JSON file.
    t0 = time.time()
    print("Parsing test JSON file...")
    test_records = parse_dataset_json(test_json, config, is_train_ds=False)
    print(f"[Test Step] Parsed test records in {time.time() - t0:.2f} seconds.")
    
    # Preprocess test images (or load saved records if available).
    t0 = time.time()
    print("Preprocessing test images (or loading saved records)...")
    test_records_updated, _ = preprocess_and_save_images(test_records, config, "test")
    print(f"[Test Step] Preprocessed test images in {time.time() - t0:.2f} seconds.")
    
    # Build the TensorFlow dataset for test images.
    t0 = time.time()
    print("Building TensorFlow dataset for test records...")
    test_ds = build_tf_dataset_from_preprocessed(test_records_updated, config)
    print(f"[Test Step] Built test dataset in {time.time() - t0:.2f} seconds.")
    
    # Evaluate the trained model on the test dataset.
    t0 = time.time()
    print("Evaluating the trained model on the test dataset...")
    results = trained_model.evaluate(test_ds, verbose=1)
    print(f"Test evaluation metrics: {results}")
    print(f"[Test Step] Model evaluation completed in {time.time() - t0:.2f} seconds.")
    
    overall_time = time.time() - overall_start
    print(f"Overall training and testing process took {overall_time:.2f} seconds.")

if __name__ == '__main__':
    main()
