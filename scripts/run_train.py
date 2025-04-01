import os
import tensorflow as tf

from src.config import Config
from src.data_pipeline import parse_dataset_json, build_tf_dataset
from src.model import create_model        
from src.train import train_transfer_model

def main():
    # 1. Load configuration.
    config = Config() 
    train_json = os.path.join(config.DATA_DIR, config.ANNOTATION_FILE, config.TRAIN_JSON_NAME)
    val_json = os.path.join(config.DATA_DIR, config.ANNOTATION_FILE, config.VAL_JSON_NAME)
    # image_path = os.path.join(config.DATA_DIR, config.IMAGE_DIR)
    # 2. Load records from JSON files.
    train_records = parse_dataset_json(train_json, config, is_train_ds=True)
    val_records = parse_dataset_json(val_json, config, is_train_ds=False)
    
    # 3. Build the TensorFlow dataset.
    train_ds = build_tf_dataset(train_records, config)
    val_ds = build_tf_dataset(val_records, config)
    
    # 4. Create your model.
    my_model = create_model(config, 'transfer', trainable_base = False)
    
    # 5. Train the model.
    h1, h2 = train_transfer_model(my_model, train_ds, val_ds, config, False)

if __name__ == '__main__':
    main()
