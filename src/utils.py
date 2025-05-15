# utils.py

import os
import random
import logging
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Optional
from dataclasses import asdict
import re

try:
    import yaml
except ImportError:
    yaml = None

def set_seed(seed_value: int = 42) -> None:
    """
    Fix the random seed for reproducibility across 
    Python's built-in random, NumPy, and TensorFlow operations.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def setup_logger(
    name: str = "tooth_classification_logger",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with a specified name and optional file output.
    Returns the logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def now_timestamp() -> str:
    """
    Returns the current date-time as a string,
    e.g. '2025-03-19_14-30-10'.
    Useful for naming log files or model outputs.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file and returns as a dictionary.
    If YAML isn't installed or file isn't YAML, you can adapt to JSON, etc.
    """
    if yaml is None:
        raise ImportError("PyYAML is not installed. Install via: pip install pyyaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Creates a timestamped directory under `base_dir` 
    to store logs, checkpoints, or other run artifacts.
    Returns the path to the newly created directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    experiment_name = now_timestamp()
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def count_trainable_params(model: tf.keras.Model) -> int:
    """
    Returns the total number of trainable parameters in a Keras model.
    """
    return np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables])

def log_config(config, path):
    """
    Logs the configuration parameters to a text file in the same directory as specified by config.LOG_DIR.
    """
    # Ensure the log directory exists.
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.txt")
    
    # Convert the dataclass to a dictionary.
    config_dict = asdict(config)
    
    # Write the config to the file in a pretty format.
    with open(config_path, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

# def convert_to_serializable(o):
#     # Convert numpy float32/float64 to Python float
#     if isinstance(o, (np.float32, np.float64)):
#         return float(o)
#     # Optionally handle numpy arrays, etc.
#     raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def convert_to_serializable(o):
    """
    Converts non-serializable types (like NumPy arrays or scalars)
    to types that the json module can handle.
    """
    if isinstance(o, np.ndarray):
        # Convert NumPy arrays to Python lists
        return o.tolist()
    elif isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
        # Convert NumPy integers to Python integers
        return int(o)
    elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
        # Convert NumPy floats to Python floats
        return float(o)
    elif isinstance(o, (np.complex_, np.complex64, np.complex128)):
        # Convert NumPy complex numbers to a serializable format (e.g., list [real, imag])
        return [o.real, o.imag]
    elif isinstance(o, (np.bool_)):
        # Convert NumPy booleans to Python booleans
        return bool(o)
    elif isinstance(o, (np.void)):
        # Handle NumPy void types if necessary (might need custom logic)
        # For now, let's raise an error or return None/string representation
        raise TypeError(f"Object of type {o.__class__.__name__} (np.void) is not handled")
    # Keep the original error for types this function doesn't explicitly handle
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def log_history(history, log_dir="logs", file_name="history.txt"):
    """
    Logs the training history to a text file in JSON format.
    
    Parameters:
        history: A History object returned by model.fit(), which contains a `history` attribute.
        log_dir (str): The directory where the history file will be saved.
        file_name (str): The name of the file to save the history.
    """
    # Ensure the log directory exists.
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, file_name)
    
    # Extract the history dictionary.
    history_dict = history.history
    
    # Write the history to a file as JSON.
    with open(file_path, "w") as f:
        json.dump(history_dict, f, indent=4, default=convert_to_serializable)
        
        
def set_trainable_layers(model: tf.keras, fine_tune_at=None):
    """
    Sets trainable layers in a model with a base model and head block.
    Handles cases where the base model might be nested within functional layers.
    
    Args:
        model: The full Keras model
        fine_tune_at: The layer index in the base model after which layers should be trainable
                     If None, all layers will be trainable
    """
    # First, make everything trainable
    model.trainable = True
    model.layers
    # Find the base model after preprocessing
    base_model = None
    preprocessing_found = False
    
    for layer in model.layers:
        if 'preprocessing' in layer.name.lower():
            preprocessing_found = True
            continue
        if preprocessing_found:
            base_model = layer
            break
            
    if base_model is None:
        print("Warning: Could not find base model after preprocessing layer")
        return

    print(f"Found base model: {base_model.name}")
    
    # Check if base_model is a Model or Layer with sublayers
    if hasattr(base_model, 'layers'):
        base_layers = base_model.layers
    elif hasattr(base_model, 'layer'):  # Some wrapped layers have .layer attribute
        base_layers = base_model.layer.layers if hasattr(base_model.layer, 'layers') else [base_model.layer]
    else:
        print(f"Warning: Base model {base_model.name} has no accessible layers")
        return
        
    if fine_tune_at is not None:
        # Validate fine_tune_at
        num_layers = len(base_layers)
        if fine_tune_at >= num_layers:
            print(f"Warning: fine_tune_at ({fine_tune_at}) is larger than base model layer count ({num_layers})")
            return
            
        # Set trainable flags in base model
        for i, layer in enumerate(base_layers):
            if i < fine_tune_at:
                layer.trainable = False
            else:
                layer.trainable = True
                print(f"Set layer {i} ({layer.name}) to trainable")
                
        print(f"Set layers {fine_tune_at} and higher in base model to trainable")
        
    # Ensure all layers after base model (head block) are trainable
    base_model_found = False
    for layer in model.layers:
        if layer == base_model:
            base_model_found = True
            continue
        if base_model_found:
            layer.trainable = True
            print(f"Set head block layer {layer.name} to trainable")
            
            
def set_trainable_layers_new(model: tf.keras.Model, fine_tune_at=None):
    """
    Sets trainable layers in a model while preserving the internal structure.
    
    Args:
        model: The full Keras model
        fine_tune_at: The layer index in the base model after which layers should be trainable
                     If None, all layers will be trainable
    """
    # First, make everything trainable
    model.trainable = True
    print(f"base model is supposed to be at model.layers[2]. Found layer name is : {model.layers[2].name}. is it correct?!")
    model.layers[2].trainable = True # to make sure!
    # Find the base model after preprocessing
    for layer in model.layers[2].layers[:fine_tune_at]:
        layer.trainable = False
        # print(f"{layer.name} layer in '{model.layers[2].name}' freezed.")

    print(f"####### Set layers {fine_tune_at} and higher in base model '{model.layers[2].name}', and the head to trainable")
    return model

def extract_epoch_number(filepath):
    """
    Extract epoch number from checkpoint filepath and convert to int.
    it will be used in train.py
    Example filepath: '/path/to/dir/best_pr_auc_weights-epoch_0014_prauc_0.46.h5'
    
    Args:
        filepath (str): Full path to the checkpoint file
        
    Returns:
        int: Epoch number (e.g., 14) or 0 if no match found
    """   
    filename = os.path.basename(filepath)
    
    # Find the epoch number
    match = re.search(r'epoch_(\d{4})', filename)
    if match:
        return int(match.group(1))  # Converts '0014' to 14
    return 0 