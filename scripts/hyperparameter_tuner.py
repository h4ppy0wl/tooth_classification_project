import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
import os
import argparse
import time
# import gc
# Add parent and current directories to sys.path if needed.
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)

from src.config import Config
from src.model import create_model
from src.train import compile_model, setup_callbacks, F1ScoreCallback
from src.utils import set_trainable_layers_new

from src.data_pipeline import (
    parse_dataset_json,
    preprocess_and_save_images,
    build_tf_dataset_from_preprocessed
)

class TransferLearningTuner(kt.HyperModel):
    def __init__(self, config: Config, train_ds, val_ds):
        super().__init__()
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds

    def build(self, hp):
        # Define hyperparameters using config space
        for param_name, param_config in self.config.HP_SPACE.items():
            if param_config['type'] == 'int':
                value = hp.Int(
                    param_name,
                    min_value=param_config['min_value'],
                    max_value=param_config['max_value'],
                    step=param_config['step']
                )
            elif param_config['type'] == 'float':
                value = hp.Float(
                    param_name,
                    min_value=param_config['min_value'],
                    max_value=param_config['max_value'],
                    step=param_config.get('step'),
                    sampling=param_config.get('sampling', 'linear')
                )
            
            # Set the config attribute
            setattr(self.config, param_name.upper(), value)
        
        # Define class weights of the parameter space
        neg_weight = hp.Float(
            'class_weight_neg',
            min_value=self.config.HP_SPACE['class_weight_neg']['min_value'],
            max_value=self.config.HP_SPACE['class_weight_neg']['max_value'],
            step=self.config.HP_SPACE['class_weight_neg']['step']
        )
        pos_weight = hp.Float(
            'class_weight_pos',
            min_value=self.config.HP_SPACE['class_weight_pos']['min_value'],
            max_value=self.config.HP_SPACE['class_weight_pos']['max_value'],
            step=self.config.HP_SPACE['class_weight_pos']['step']
        )
        
        self.config.CLASS_WEIGHTS = {0: neg_weight, 1: pos_weight}
        
        # Create model with current hyperparameters
        model = create_model( self.config, 'transfer', trainable_base=False)
        
        # Compile model
        model = compile_model(model, self.config, self.config.INITIAL_LR)
        
        return model

    def fit(self, hp, model, epochs = None, *args, **kwargs):
        # Initial training phase
        # the model is pre-compiled in the build method
        # so we can directly use it here
        initial_history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,#self.config.NUM_INITIAL_EPOCHS,
            class_weight= self.config.CLASS_WEIGHTS,
            callbacks=[
                        # EarlyStopping(
                        #     monitor='val_auc',
                        #     patience=7,
                        #     min_delta = 0.005,
                        #     verbose=1,
                        #     restore_best_weights=True
                        # ),
                        ReduceLROnPlateau(
                            monitor='val_auc',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-7,
                            min_delta=0.001,
                            mode='max',
                            verbose=1
                        ),
                        F1ScoreCallback(thresholds=self.config.METRIC_THRESHOLDS),
                    ],
            verbose=1
        )
        
        # Fine-tuning phase_ with current config this will not run
        if self.config.FINE_TUNE_FROM_LAYER > 0:
            mymodel = set_trainable_layers_new(mymodel, self.config.FINE_TUNE_FROM_LAYER)       
                
            # Recompile with lower learning rate
            model = compile_model(model, self.config, self.config.FINE_TUNE_LR)
            
            fine_tune_history = model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.config.NUM_FINE_TUNE_EPOCHS,
                class_weight= self.config.CLASS_WEIGHTS,
                callbacks=[
                        # EarlyStopping(
                        #     monitor='val_auc',
                        #     patience=7,
                        #     min_delta = 0.005,
                        #     verbose=1,
                        #     restore_best_weights=True
                        # ),
                        ReduceLROnPlateau(
                            monitor='val_auc',
                            factor=0.5,
                            patience=4,
                            min_lr=1e-7,
                            min_delta=0.001,
                            mode='max',
                            verbose=1
                        ),
                        F1ScoreCallback(thresholds=self.config.METRIC_THRESHOLDS),
                    ],
                verbose=1
            )
            
            # Return the best validation AUC from either training phase
            return max(
                max(initial_history.history['val_auc']),
                max(fine_tune_history.history['val_auc'])
            )
        
        return max(initial_history.history['val_auc'])

def run_hyperparameter_tuning(config: Config, train_ds, val_ds, result_path = "./"):
    from contextlib import redirect_stdout
    # Create directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    dataset_code = (
        f"{config.TARGET_CLASS}"
        f"{config.MASK_VALUE}"
        f"{config.RANDOM_SEED}"
        f"{int(config.AUGMENT_DATA)}"  # Convert bool to 0 or 1.
        f"{int(config.NORMALIZE_IMAGES)}"
        f"{int(config.MASK_BG)}"
        f"{config.DARK_IMAGE_THRESHOLD}"
        f"{config.POLYGON_SMOOTHING_TOLERANCE}"
    )
    tuner = kt.Hyperband(
        TransferLearningTuner(config, train_ds, val_ds),
        objective='val_auc',
        mode = 'max',
        max_epochs=config.NUM_INITIAL_EPOCHS + config.NUM_FINE_TUNE_EPOCHS,
        factor=3,
        directory= result_path,
        project_name=f'{config.MODEL_ARCHITECTURE}_{dataset_code}_tuning'
    )
    
    # Print search space summary
    tuner.search_space_summary()
    
    # Save search space summary
    with open(os.path.join(result_path, 'search_space_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            tuner.search_space_summary()
    
    # Perform hyperparameter search
    tuner.search()
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    
    # Save best hyperparameters
    with open(os.path.join(result_path, 'best_hyperparameters.txt'), 'w') as f:
        for param in best_hps.values:
            f.write(f"{param}: {best_hps.get(param)}\n")
    
    # Save results summary
    with open(os.path.join(result_path, 'results_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            tuner.results_summary()
    
    
    # Print results summary
    tuner.results_summary()
    
    return best_hps

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument('--result_path', type=str, default='./results',
                      help='Path to save tuning results (default: ./results)')
    args = parser.parse_args()
    

    print("###### starting hyperparameter tuning")
    # Start time for overall process
    overall_start = time.time()
    overall_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(overall_start))
    print(f"Overall start time: {overall_start_time}")  
    config = Config()


    # Construct JSON paths for train, validation, and test.
    print("###### loading jsons ")
    train_json = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.TRAIN_JSON_NAME)
    val_json   = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, config.VAL_JSON_NAME)

    # Parse records from JSON files.
    print("###### parsing jsons ")
    train_records = parse_dataset_json(train_json, config, is_train_ds=True)
    val_records   = parse_dataset_json(val_json, config, is_train_ds=False)

    # Preprocess images (or load preprocessed records if available).
    print("###### Checking/preprocessing images ")
    train_records_updated, _ = preprocess_and_save_images(train_records, config, "train")
    val_records_updated, _   = preprocess_and_save_images(val_records, config, "val")

    # Build TensorFlow datasets.
    print("###### building tf dataset ")
    train_ds = build_tf_dataset_from_preprocessed(train_records_updated, config, is_training= True)
    val_ds   = build_tf_dataset_from_preprocessed(val_records_updated, config)
    
    # Run hyperparameter tuning
    best_hps = run_hyperparameter_tuning(config, train_ds, val_ds, result_path=args.result_path)
    
    # time taken for tuning
    overall_end = time.time()
    overall_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(overall_end))
    overall_time_taken = overall_end - overall_start
    print(f"Overall end time: {overall_end_time}")
    print(f"Overall time taken: {overall_time_taken} seconds")
    # Print best hyperparameters
    print("Best hyperparameters:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")
    
if __name__ == "__main__":
    main()