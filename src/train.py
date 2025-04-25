import os
import sys
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)
from src import model as model_lib
from src.config import Config
from src.utils import log_config, log_history, set_trainable_layers_new
import numpy as np

class F1ScoreCallback(tf.keras.callbacks.Callback):
    """
    A Keras callback to calculate and log F1 score based on precision and recall
    metrics computed *at specific thresholds* during training.

    Expects the model to be compiled with tf.keras.metrics.Precision(thresholds=...)
    and tf.keras.metrics.Recall(thresholds=...) metrics.
    """
    def __init__(self, thresholds, **kwargs):
        """
        Initializes the callback.

        Args:
            thresholds: The float or list of floats used for the Precision/Recall metrics.
            **kwargs: Additional keyword arguments for the base Callback class.
        """
        super().__init__(**kwargs)
        # Ensure thresholds is always a list for consistent processing
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
        if not all(isinstance(t, (float, int)) for t in self.thresholds):
             raise ValueError("Thresholds must be a float or a list of floats/integers.")

    def _calculate_f1(self, p, r):
        """Safely calculates F1 score, handling potential division by zero."""
        # Ensure p and r are numpy arrays for element-wise operations
        p = np.array(p)
        r = np.array(r)
        # Use numpy's divide function for safe division with a 'where' clause
        denominator = p + r
        f1 = np.divide(2 * p * r, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator!=0)
        return f1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for threshold in self.thresholds:
            # Construct the names of the metrics Keras logs
            # Ensure these names match exactly how you defined them in model.compile()
            precision_key = f'precision_at_{threshold}'
            recall_key = f'recall_at_{threshold}'
            f1_key = f'f1_at_{threshold}'
            val_precision_key = f'val_{precision_key}'
            val_recall_key = f'val_{recall_key}'
            val_f1_key = f'val_{f1_key}'

            # --- Calculate Training F1 ---
            if precision_key in logs and recall_key in logs:
                p = logs[precision_key]
                r = logs[recall_key]

                # If Precision/Recall metrics return lists (for multiple thresholds),
                # find the value corresponding to the current threshold.
                # This usually happens if the *metrics* were defined with a list,
                # even if the callback processes one threshold at a time.
                # Note: Keras might already flatten this if only one threshold used,
                # but this handles the case where metrics have lists.
                if isinstance(p, (list, np.ndarray)):
                   try:
                       # Assuming the metric outputs align with self.thresholds order
                       idx = self.thresholds.index(threshold)
                       p = p[idx]
                       r = r[idx]
                   except (ValueError, IndexError):
                       print(f"\nWarning: Could not find index for threshold {threshold} in metric results for {precision_key}. Skipping F1 calculation.")
                       continue # Skip F1 calc if index mapping fails


                f1_train = self._calculate_f1(p, r)
                logs[f1_key] = f1_train # Add to logs
                # Optional: print if you want immediate feedback besides the progress bar
                print(f" - {f1_key}: {f1_train:.4f}", end="")

            # --- Calculate Validation F1 ---
            if val_precision_key in logs and val_recall_key in logs:
                p_val = logs[val_precision_key]
                r_val = logs[val_recall_key]

                # Handle potential list output from metrics
                if isinstance(p_val, (list, np.ndarray)):
                   try:
                       idx = self.thresholds.index(threshold)
                       p_val = p_val[idx]
                       r_val = r_val[idx]
                   except (ValueError, IndexError):
                       print(f"\nWarning: Could not find index for threshold {threshold} in metric results for {val_precision_key}. Skipping val F1 calculation.")
                       continue # Skip val F1 calc

                f1_val = self._calculate_f1(p_val, r_val)
                logs[val_f1_key] = f1_val # Add to logs
                # Optional: print
                print(f" - {val_f1_key}: {f1_val:.4f}", end="")

        # Optional: Add a newline if you were printing custom messages
        # if any(f'f1_at_{t}' in logs or f'val_f1_at_{t}' in logs for t in self.thresholds):
        #    print() # Add a newline after printing custom metrics

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="balanced_accuracy", **kwargs):
        """
        Initializes the BalancedAccuracy metric.

        Args:
            threshold (float): The threshold to convert probabilities to binary predictions.
            name (str): Name of the metric instance.
            **kwargs: Additional keyword arguments.
        """
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        # Create variables to store true positives, true negatives, false positives, false negatives.
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state variables with the values computed on the current batch.

        Args:
            y_true (Tensor): The ground truth labels.
            y_pred (Tensor): The predicted probabilities.
            sample_weight (Optional[Tensor]): Optional weights for scaling each example.
        """
        # Binarize predictions using the given threshold.
        y_pred_binary = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate metrics for the current batch.
        tp = tf.reduce_sum(y_true * y_pred_binary)
        fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
        fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
        
        # Update the accumulated metric values.
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
    
    def result(self):
        """
        Computes the balanced accuracy, which is the average of sensitivity and specificity.

        Returns:
            A scalar tensor representing the balanced accuracy.
        """
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        return 0.5 * (sensitivity + specificity)
    
    def reset_states(self):
        """Resets all of the metric state variables."""
        self.true_positives.assign(0)
        self.false_negatives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)


def compile_model(model, config: Config, learning_rate: float) -> tf.keras.Model:
    """
    Compiles the model with specified parameters, loss function, and metrics.
    
    Args:
        model: The Keras model to compile
        config: Configuration object containing model parameters
        learning_rate: Learning rate for the optimizer
        
    Returns:
        The compiled model
    """
    #Setting the optimizer
    if config.OPTIMIZER.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum= config.MOMENTUM)
    
    elif config.OPTIMIZER.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum= config.MOMENTUM)
    
    elif config.OPTIMIZER.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#, clipnorm = 1.0)

    elif config.OPTIMIZER.lower() == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    elif config.OPTIMIZER.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adamgrad(learning_rate=learning_rate)

    #Setting the loss functionwhat are my choices regarding loa
    if config.LOSS_FUNC.lower() == 'binaryfocalcrossentropy':
        loss_func = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,
                                                            gamma=config.BFC_GAMMA,
                                                            from_logits=False, 
                                                            label_smoothing=0.0,
                                                            reduction="sum_over_batch_size",
                                                            name="binary_focal_crossentropy",
                                                            )
    
    elif config.LOSS_FUNC.lower() == 'binarycrossentropy':
        loss_func = tf.keras.losses.BinaryCrossentropy( from_logits=False, 
                                                        label_smoothing=0.0,
                                                        reduction="auto",
                                                        name="binary_crossentropy",
                                                        )

    model.compile(
        optimizer= optimizer,
        loss= loss_func,
        metrics=[
            tf.keras.metrics.Precision(
                name=f'precision_at_{config.METRIC_THRESHOLDS[0]}', 
                thresholds=config.METRIC_THRESHOLDS
            ),
            tf.keras.metrics.Recall(
                name=f'recall_at_{config.METRIC_THRESHOLDS[0]}', 
                thresholds=config.METRIC_THRESHOLDS
            ),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc',
                curve='PR',        # Precision–Recall curve
                summation_method='interpolation'  # (default) gives a smooth estimate
            ),
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy",
                threshold=config.METRIC_THRESHOLDS[0]
            ),
        ]
    )
    return model

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch} validation data sample:")
        for x_batch, y_batch in self.model.validation_data.take(1):
            print(f"X shape: {x_batch.shape}")
            print(f"Y shape: {y_batch.shape}")
            print(f"Y values: {y_batch.numpy()}")
            
class GradientDebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if batch % 100 == 0:  # Check every 100 batches
            with tf.GradientTape() as tape:
                x_batch = self.model.inputs[0]
                y_batch = self.model.targets[0]
                predictions = self.model(x_batch, training = True)
                loss_value = self.model.compiled_loss(y_batch, predictions)

            weights = self.model.trainable_weights
            grads = tape.gradients(loss_value, weights)
            grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in grads]
            print(f"\nBatch {batch} gradient norms: {grad_norms}")

def setup_callbacks(config: Config, log_dir: str) -> list:
    """Setup callbacks for initial training phase"""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq='epoch',
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )
    
    checkpoint_path = os.path.join(log_dir, "cp-{epoch:04d}_{val_auc:.2}.h5")

    f1_callback = F1ScoreCallback(thresholds=config.METRIC_THRESHOLDS)
    

    return [
        # EarlyStopping(
        #     monitor='val_pr_auc',
        #     patience=5,
        #     min_delta = 0.005,
        #     verbose=1,
        #     restore_best_weights=True
        # ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            min_delta=0.005,
            mode='max',
            verbose=1
        ),
        tensorboard_callback,
        f1_callback,
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_path,
        #     save_weights_only=True,
        #     save_freq='epoch',
        #     save_best_only=False,
        #     verbose=1
        # ),
        # DebugCallback(),
        # GradientDebugCallback(),
    ]

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
    c_weights = config.CLASS_WEIGHTS

    # Create a log directory with a timestamp.
    log_tail_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = config.LOG_DIR +"/tensorboard/"+ config.MODEL_ARCHITECTURE +"/"+ log_tail_path
    log_config(config, log_dir)
    # Instantiate the TensorBoard callback.
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir,
    #     update_freq='epoch',
    #     histogram_freq=0,      # Frequency (in epochs) at which to compute activation and weight histograms.
    #     write_graph=True,      # Whether to visualize the graph in TensorBoard.
    #     write_images=True      # Whether to save model weights as images.
    # )
    
    
    # Initial training setup
    initial_callbacks = setup_callbacks(config, log_dir)
    
    # Phase 1: Initial training with frozen base
    mymodel = compile_model(mymodel, config, initial_lr)
    
    
    print("**************** Starting initial training **************** ")
    trainable_param_count = np.sum([tf.keras.backend.count_params(w) for w in mymodel.trainable_weights])
    print("trainable params in initial model:", trainable_param_count)
    initial_history = mymodel.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=i_epochs,
        class_weight= c_weights,
        callbacks=initial_callbacks,
        verbose = 1,
    )

    initial_history.history['trainable_params'] = trainable_param_count
    completed_epochs = len(initial_history.history['loss'])
    log_history(initial_history, log_dir, f"initial_training_history_{num}.json")
    # Save weights after initial training.
    # if config.NUM_FINE_TUNE_EPOCHS == 0:
    if save_models:
        path = os.path.join(log_dir,f"{initial_weights_name}_{num}_.h5")
        mymodel.save_weights(path)
        print(f"Initial model weights saved to: {path}")
        fine_tune_history = {}
    # else:
    if config.NUM_FINE_TUNE_EPOCHS > 0:
        # Phase 2: Fine-tuning.
        print("************ Fine-tuning model **************** ")
        print(f" **** Start fine tuning from epoch {completed_epochs +1}")
        
        print("Model trainable weights before fine-tuning:", len(mymodel.trainable_weights))
        
        # Unfreeze the base model and set the layers to be trainable.
        # if fine_tune_at > 0:
        mymodel = set_trainable_layers_new(mymodel, fine_tune_at)

        fine_tune_callbacks = setup_callbacks(config, log_dir)
        mymodel = compile_model(mymodel, config, fine_tune_lr)
        trainable_param_count = np.sum([tf.keras.backend.count_params(w) for w in mymodel.trainable_weights])
        print("trainable params in fine tuning model:", trainable_param_count)


        fine_tune_history = mymodel.fit(
            train_dataset,
            validation_data=val_dataset,
            initial_epoch= completed_epochs,
            epochs= completed_epochs + fine_tune_epochs,
            class_weight= c_weights,
            callbacks=fine_tune_callbacks,
            verbose = 1
        )

        fine_tune_history.history['trainable_params'] = trainable_param_count
        # Save weights after fine-tuning.
        mymodel.trainable = False
        if save_models:
            path = os.path.join(log_dir,f"{fine_tuned_weights_name}_{num}_.h5")
            mymodel.save_weights(path)
            print(f"Fine tuned model weights saved to: {path}")
        # fine_tune_history = initial_history # for test
        log_history(fine_tune_history, log_dir, f"fine_tune_history_{num}.json")
    
    return mymodel , initial_history, fine_tune_history, log_dir