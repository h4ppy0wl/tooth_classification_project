import os
import sys
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)
from src import model as model_lib
from src.config import Config
from src.utils import log_config, log_history


def train_attention_model( config: Config,
                            attention_model,
                            train_ds,
                            val_ds):
    # 2) Compile with dictionary losses
    # - classification_output uses standard binary crossentropy
    # - attention_output also uses binary crossentropy 
    #   (we want the attention map to match the mask)
    losses = ["binary_crossentropy","binary_crossentropy"]#{"classification_output": "binary_crossentropy","attention_output": "binary_crossentropy"}
    loss_weights = [1.0, 1.0]#{"classification_output": 1.0,"attention_output": 1.0}

    attention_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={"classification_output": "accuracy"}  # or any relevant metrics
    )

    # 3) Prepare your data.
    # Suppose you have:
    #  - X_train: shape (num_samples, 224, 224, 3)
    #  - y_class_train: shape (num_samples,)  => 0/1 for no-plaque/plaque
    #  - y_mask_train: shape (num_samples, H', W', 1) => your plaque masks or 0 for no-plaque
    #
    # If your model's attention_output is shape (batch, H, W, 1),
    # you must ensure y_mask_train also matches (H, W, 1).
    # If not, you can either:
    #   1) resize your masks to match the final shape,
    #   2) or upsample/downsample them in a pipeline,
    #   3) or adapt the model to produce the same resolution as your masks.
    #
    # For demonstration, let's assume we've aligned them to the final conv shape.

    # We'll do a direct .fit(...) with a dict of outputs:
    # model.fit(X_train, {"classification_output": y_class_train,
    #                    "attention_output": y_mask_train},
    #           batch_size=16,
    #           epochs=10,
    #           validation_data=(X_val, {"classification_output": y_class_val,
    #                                   "attention_output": y_mask_val})
    # )

    print(attention_model.summary())
    return attention_model




def train_transfer_model(mymodel, train_dataset, val_dataset, config: Config, save_models = True,
                        initial_weights_name="initial_weights",
                        fine_tuned_weights_name="fine_tuned_weights",
                        num = 1):
    """
    Train a transfer learning model in two phases: initial training (with a frozen base) then fine-tuning.
    Early stopping is applied based on both training loss and validation loss.

    Args:
        model (tf.keras.Model): The transfer model.
        train_dataset: tf.data.Dataset for training.
        val_dataset: tf.data.Dataset for validation.
        initial_epochs int: Epochs when the base model is frozen.
        fine_tune_epochs (int): Epochs for fine-tuning.
        initial_lr (float): Learning rate for initial training.
        fine_tune_lr (float): Learning rate for fine-tuning.
        base_model_prefix (str): Optional prefix to identify the base model.
        fine_tune_at (int): If provided, only layers after this index in the base model will be unfrozen.
        initial_weights_path (str): File path to save weights after initial training.
        fine_tuned_weights_path (str): File path to save weights after fine-tuning.
        num: is used when this function is run in a loop and weights and reports are going to be saved

    Returns:
        Tuple: (initial_history, fine_tune_history)
    """
    i_epochs = config.NUM_INITIAL_EPOCHS
    fine_tune_epochs=config.NUM_FINE_TUNE_EPOCHS
    initial_lr=config.INITIAL_LR
    fine_tune_lr=config.FINE_TUNE_LR
    fine_tune_at=config.FINE_TUNE_FROM_LAYER

    # Create a log directory with a timestamp.
    log_tail_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = config.LOG_DIR +"/tensorboard/"+ config.MODEL_ARCHITECTURE +"/"+ log_tail_path
    log_config(config, log_dir)
    # Instantiate the TensorBoard callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq='epoch',
        histogram_freq=0,      # Frequency (in epochs) at which to compute activation and weight histograms.
        write_graph=True,      # Whether to visualize the graph in TensorBoard.
        write_images=True      # Whether to save model weights as images.
    )
    
    # Define callbacks for the initial training phase:
    # ReduceLROnPlateau here will monitor validation loss and reduce LR if no improvement
    initial_callbacks = [
        # EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        tensorboard_callback
    ]
    
    # Phase 1: Initial training with frozen base.
    mymodel.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss= BinaryFocalCrossentropy(
                apply_class_balancing=True,
                # alpha=0.25,
                gamma=2.0,
                from_logits=False,
                label_smoothing=0.0,
                reduction="sum_over_batch_size",
                name="binary_focal_crossentropy"
                ),
        metrics=[
            # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(name='f1score', average = 'weighted')
        ]
    )
    
    print("**************** Starting initial training **************** ")
    initial_history = mymodel.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=i_epochs,
        callbacks=initial_callbacks,
        verbose = 1,
    )
    
    log_history(initial_history, log_dir, f"initial_training_history_{num}.json")
    # Save weights after initial training.
    if config.NUM_FINE_TUNE_EPOCHS == 0:
        if save_models:
            path = os.path.join(log_dir,f"{initial_weights_name}_{num}_.h5")
            mymodel.save_weights(path)
            print(f"Initial model weights saved to: {path}")
    else:
        # Phase 2: Fine-tuning.
        print("************ Fine-tuning model **************** ")
        # Unfreeze all layers initially.
        mymodel.trainable = True

        if fine_tune_at is not None:
            for i, layer in enumerate(mymodel.layers):
                if i < fine_tune_at:
                    layer.trainable = False
                else:
                    layer.trainable = True
            print(f"Layer {fine_tune_at} and higher set to trainable in fine tuning step.")

        # Define callbacks for the fine-tuning phase:
        fine_tune_callbacks = [
            # EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor='val_loss',  # You can also set this to 'loss' if you prefer
                factor=0.5,          # Factor by which the LR will be reduced
                patience=2,          # Number of epochs with no improvement after which LR is reduced
                min_lr=1e-7,         # Lower bound on the learning rate
                verbose=1
            ),
            tensorboard_callback
        ]
        
        # Recompile with a lower learning rate.
        mymodel.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss=BinaryFocalCrossentropy(
                    apply_class_balancing=True,
                    # alpha=0.25,
                    gamma=2.0,
                    from_logits=False,
                    label_smoothing=0.0,
                    reduction="sum_over_batch_size",
                    name="binary_focal_crossentropy"
                    ),
            metrics=[
                # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1score', average = 'weighted')
            ]
        )

        fine_tune_history = mymodel.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            verbose = 1
        )

        # Save weights after fine-tuning.
        if save_models:
            path = os.path.join(log_dir,f"{fine_tuned_weights_name}_{num}_.h5")
            mymodel.save_weights(path)
            print(f"Fine tuned model weights saved to: {path}")
        
        log_history(fine_tune_history, log_dir, f"fine_tune_history_{num}.json")
    
    return mymodel , initial_history, fine_tune_history, log_dir