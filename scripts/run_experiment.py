import os
import sys
import csv
import io
import time
import contextlib
import tensorflow as tf
import gc

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

def run_experiment(config, exp_num):
    """
    Runs the training and evaluation pipeline for the given config.
    Returns a dictionary with experiment details.
    """
    overall_start = time.time()

    # Construct JSON paths for train, validation, and test.
    train_json = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.TRAIN_JSON_NAME)
    val_json   = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.VAL_JSON_NAME)
    test_json  = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.TEST_JSON_NAME)

    # Parse records from JSON files.
    train_records = parse_dataset_json(train_json, config, is_train_ds=True)
    val_records   = parse_dataset_json(val_json, config, is_train_ds=False)
    test_records  = parse_dataset_json(test_json, config, is_train_ds=False)

    # Preprocess images (or load preprocessed records if available).
    train_records_updated, data_folder_name = preprocess_and_save_images(train_records, config, "train")
    val_records_updated, _   = preprocess_and_save_images(val_records, config, "val")
    test_records_updated, _  = preprocess_and_save_images(test_records, config, "test")

    # Build TensorFlow datasets.
    train_ds = build_tf_dataset_from_preprocessed(train_records_updated, config)
    val_ds   = build_tf_dataset_from_preprocessed(val_records_updated, config)
    test_ds  = build_tf_dataset_from_preprocessed(test_records_updated, config)

    # Create the model.
    model = create_model(config, 'transfer', trainable_base=False)

    # Train the model.
    start_train = time.time()
    # Here, train_transfer_model is expected to return:
    # (trained_model, initial_history, fine_tune_history, current_log_dir)
    trained_model, h1, h2, current_log_dir = train_transfer_model(model, train_ds, val_ds, config, save_models=True, num=exp_num)
    train_time = time.time() - start_train

    # Evaluate the model on the test dataset.
    start_eval = time.time()
    test_results = trained_model.evaluate(test_ds, verbose=1)
    eval_time = time.time() - start_eval

    overall_time = time.time() - overall_start
    
    try:
        print("freeup memory to prevent memory leak")
        del model
        del trained_model
        del h1
        del h2
        del train_ds
        del val_ds
        del test_ds
        collected = gc.collect()
        tf.keras.backend.clear_session()
        print("Garbage collector: collected",
                "%d objects." % collected)
        print("cleaned!")
    except Exception as e:
        print("An error occured during memory releasing: ", e)


    # Return all experiment details in a dictionary.
    return {
        "current_log_dir": current_log_dir,
        "data_folder_name": data_folder_name,
        "weight_file": os.path.join(current_log_dir,f"initialorfinetuned_weights_{exp_num}_.h5"),  # adjust if needed
        "initial_history_file": os.path.join(current_log_dir,f"initial_training_history_{exp_num}.json"),
        "fine_tune_history_file": os.path.join(current_log_dir,f"fine_tune_history_{exp_num}.json"),
        "train_time": train_time,
        "eval_time": eval_time,
        "overall_time": overall_time,
        "test_results": test_results,
    }

def main():
    # Define a list of parameter combinations for experiments.
    # Adjust these combinations as needed.
    experiment_params = [
        {"MASK_VALUE": 128, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128},
        {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256},
        {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": False, "MASK_BG": True, "HEAD_DENSE_UNITS": 256},
    ]

    # CSV file to collect results.
    csv_file = Config.LOG_DIR +"/tensorboard/"+ Config.MODEL_ARCHITECTURE +"/experiment_results.csv"
    # csv_file = "experiment_results.csv"
    csv_fields = [
        "experiment_id","data_folder_name","test_results","HEAD_DENSE_UNITS","MASK_VALUE","AUGMENT_DATA",
        "NORMALIZE_IMAGES","MASK_BG","current_log_dir","train_time","eval_time","overall_time",
        "weight_file","initial_history_file","fine_tune_history_file"
    ]

    # Open CSV file for writing.
    with open(csv_file, mode="w", newline="") as f_csv:
        csv_writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        csv_writer.writeheader()

        # Loop over each parameter combination.
        exp_i = 1
        for params in experiment_params:
            # Create a new config instance and update it with the current parameter set.
            config = Config()
            for key, value in params.items():
                setattr(config, key, value)

            # # Capture all terminal output for this experiment.
            # log_capture = io.StringIO()
            # with contextlib.redirect_stdout(log_capture):
            result = run_experiment(config, exp_num=exp_i)
            # logs = log_capture.getvalue()

            # # Save the captured logs into the current experiment log folder.
            # experiment_log_dir = result["current_log_dir"]
            # os.makedirs(experiment_log_dir, exist_ok=True)
            # log_file_path = os.path.join(experiment_log_dir, "experiment_log.txt")
            # with open(log_file_path, "w") as log_file:
            #     log_file.write(logs)

            # Use the last folder name (basename) of current_log_dir as experiment id.
            experiment_id = os.path.basename(os.path.normpath(result["current_log_dir"]))

            # Prepare a row for the CSV file.
            result_row = {
                "experiment_id": experiment_id,
                "data_folder_name": result["data_folder_name"],
                "test_results": result["test_results"],
                'HEAD_DENSE_UNITS': params.get("HEAD_DENSE_UNITS"),
                "MASK_VALUE": params.get("MASK_VALUE"),
                "AUGMENT_DATA": params.get("AUGMENT_DATA"),
                "NORMALIZE_IMAGES": params.get("NORMALIZE_IMAGES"),
                "MASK_BG": params.get("MASK_BG"),
                "current_log_dir": result["current_log_dir"],
                "train_time": result["train_time"],
                "eval_time": result["eval_time"],
                "overall_time": result["overall_time"],
                "weight_file": result["weight_file"],
                "initial_history_file": result["initial_history_file"],
                "fine_tune_history_file": result["fine_tune_history_file"],
            }
            csv_writer.writerow(result_row)
            f_csv.flush()

            print(f"Experiment {experiment_id} completed. Results saved.")
            exp_i +=1



if __name__ == '__main__':
    main()
