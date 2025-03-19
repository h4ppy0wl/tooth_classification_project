# utils.py

import os
import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Optional

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

def early_stop_callback(
    monitor: str = "val_loss", 
    patience: int = 5
) -> tf.keras.callbacks.EarlyStopping:
    """
    Returns a Keras EarlyStopping callback configured with 
    the given monitor metric and patience.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True
    )
