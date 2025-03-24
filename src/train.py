from src import model
from src.config import Config
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_transfer_model(mymodel, train_dataset, val_dataset,
                        initial_epochs=Config.NUM_INITIAL_EPOCHS,
                        fine_tune_epochs=Config.NUM_FINE_TUNE_EPOCHS,
                        initial_lr=Config.INITIAL_LR,
                        fine_tune_lr=Config.FINE_TUNE_LR,
                        fine_tune_at=Config.FINE_TUNE_FROM_LAYER,
                        initial_weights_path="initial_weights.h5",
                        fine_tuned_weights_path="fine_tuned_weights.h5"):
    """
    Train a transfer learning model in two phases: initial training (with a frozen base) then fine-tuning.
    Early stopping is applied based on both training loss and validation loss.

    Args:
        model (tf.keras.Model): The transfer model.
        train_dataset: tf.data.Dataset for training.
        val_dataset: tf.data.Dataset for validation.
        initial_epochs (int): Epochs when the base model is frozen.
        fine_tune_epochs (int): Epochs for fine-tuning.
        initial_lr (float): Learning rate for initial training.
        fine_tune_lr (float): Learning rate for fine-tuning.
        base_model_prefix (str): Optional prefix to identify the base model.
        fine_tune_at (int): If provided, only layers after this index in the base model will be unfrozen.
        initial_weights_path (str): File path to save weights after initial training.
        fine_tuned_weights_path (str): File path to save weights after fine-tuning.

    Returns:
        Tuple: (initial_history, fine_tune_history)
    """
    
    # Define callbacks for the initial training phase:
    initial_callbacks = [
        EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    ]
    
    # Phase 1: Initial training with frozen base.
    mymodel.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("Starting initial training...")
    initial_history = mymodel.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epochs,
        callbacks=initial_callbacks
    )
    
    # Save weights after initial training.
    mymodel.save_weights(initial_weights_path)
    print(f"Initial model weights saved to: {initial_weights_path}")
    
    # Define callbacks for the fine-tuning phase:
    fine_tune_callbacks = [
        EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    ]
    
    # Phase 2: Fine-tuning.
    print("Fine-tuning model...")
    # Unfreeze all layers initially.
    mymodel.trainable = True

    if fine_tune_at is not None:
        for i, layer in enumerate(mymodel.layers):
            if i < fine_tune_at:
                layer.trainable = False
            else:
                layer.trainable = True

    # Recompile with a lower learning rate.
    mymodel.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    fine_tune_history = mymodel.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=fine_tune_epochs,
        callbacks=fine_tune_callbacks
    )

    # Save weights after fine-tuning.
    mymodel.save_weights(fine_tuned_weights_path)
    
    return initial_history, fine_tune_history