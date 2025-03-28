# model.py
import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from src.config import Config


def build_pretrained_model(architecture, 
                            input_shape: tuple[int,int,int],
                            trainable_base: bool = False,
                            fine_tune_at: int = None):
    """
    architecture: str, one of {'resnet50','inceptionv3','efficientnetb0', ...}
    freeze: whether to freeze base model layers initially
    Returns a compiled model
    """
    architectures = ['ResNet50', 'InceptionV3', 'EfficientNetB0', 'EfficientNetV2B0', 'EfficientNetV2B1', 'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtMedium', 'ConvNeXtLarge']
    
    if architecture.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)
        preprocess_func = resnet_preprocess

    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {architectures}")

    
    
    # We freeze the base model in the first step to prevent weight deterioration
    if not trainable_base:
        # Freeze entire base model for initial training
        base_model.trainable = False
    else:
        if fine_tune_at is not None:
            # Freeze layers until fine_tune_at and unfreeze from that point onward
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        else:
            base_model.trainable = True

    # Create an Input layer and add a Lambda layer for preprocessing.
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(preprocess_func, name="preprocessing_layer")(inputs)
    
    # Pass preprocessed inputs through the base model.
    features = base_model(x)


    # Add a custom classification head
    # option 1: GlobalAveragePooling2D + Dense128 + Dropout02 + Dense
    # x = layers.GlobalAveragePooling2D()(base_model.output)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(Config.DROPOUT_RATE)(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    
    # option 2: GlobalAveragePooling2D + BN + Dense256 + Dropout05 + Dense
    gap = layers.GlobalAveragePooling2D()(features)
    bn = layers.BatchNormalization()(gap)  # helps with feature scale
    dns = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(bn)
    do = layers.Dropout(Config.DROPOUT_RATE)(dns)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(do)
    
    # option 3: Flatten + Dense256 + Dropout05 + Dense
    # x = layers.Flatten()(x)  # Flatten the spatial feature maps
    # x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    # x = layers.Dropout(Config.DROPOUT_RATE)(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    

    model = Model(inputs=inputs, outputs=classification_output, name=f'{architecture}_model')



    return model

def build_pretrained_attention_model(architecture: str,
                                        input_shape: tuple[int,int,int],
                                        trainable_base: bool = False,
                                        fine_tune_at: int = None
                                    ) -> tf.keras.Model:
    """
    Build a transfer-learning model with a built-in attention mechanism.
    It produces two outputs:
        1) classification_output (sigmoid)
        2) attention_output (spatial attention map)
    
    During training:
      - You will feed class labels to 'classification_output'
      - You will feed plaque masks (or zeros) to 'attention_output'

    At inference time:
      - You can ignore 'attention_output' and use only 'classification_output'.

    Parameters:
        architecture: str, one of {'resnet50','inceptionv3','efficientnetb0', ...}
        input_shape: (H, W, C)
        trainable_base: whether to unfreeze (train) the base model's layers
        fine_tune_at: layer index from which to unfreeze. If None and trainable_base=True, unfreezes entire base model.

    Returns:
        Keras functional Model with 2 outputs. 
    """
    # ------------------
    # 1) Choose Backbone
    # ------------------
    architectures = ['resnet50','inceptionv3','efficientnetb0']
    arch_lower = architecture.lower()
    
    if arch_lower == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_func = resnet_preprocess
    else:
        raise ValueError(f"Unknown architecture {architecture}. Choose from {architectures}")

    # ----------------------
    # 2) Freeze / Fine-tune
    # ----------------------
    if not trainable_base:
        # Freeze entire base model
        base_model.trainable = False
    else:
        if fine_tune_at is not None:
            # Freeze up to the specified layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        else:
            # Unfreeze entire base model
            base_model.trainable = True

    # -----------------------------
    # 3) Build the Attention Branch
    # -----------------------------
    # We'll do it in the functional style

    # Input + Preprocessing
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    x = layers.Lambda(preprocess_func, name="preprocessing_layer")(inputs)

    # Pass through pretrained backbone
    features = base_model(x)  # shape ~ (batch, H, W, C)

    # Attention map (1 channel)
    attention_logits = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),  # optional
        name='attention_conv'
    )(features)

    attention_output = layers.Activation('sigmoid', name='attention_output')(attention_logits)
    # shape ~ (batch, H, W, 1), range [0..1]

    # Multiply features by attention
    weighted_features = layers.Multiply(name='apply_attention')([features, attention_output])

    # Global pooling & classification
    gap = layers.GlobalAveragePooling2D(name='global_average_pool')(weighted_features)
    # You can add dropout or dense layers here if desired
    bn = layers.BatchNormalization()(gap)  # helps with feature scale
    dns = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(bn)
    do = layers.Dropout(Config.DROPOUT_RATE, name='gap_dropout')(dns)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(do)

    # ---------------------
    # 4) Create 2-output Model
    # ---------------------
    model = Model(inputs=inputs, outputs=[classification_output, attention_output],
                    name=f'{architecture}_attention_model')

    return model

def build_simple_cnn(
    input_shape: tuple[int,int,int],
    base_filters=32,
) -> tf.keras.Model:
    """
    Build a simple CNN for classification.
    - input_shape: The shape of input images (H, W, C)
    - num_classes: Number of output classes (e.g., 2 for plaque vs. no_plaque)
    - base_filters: Number of filters in the initial conv layer (can expand deeper in the network)
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional block 1
    x = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional block 2
    x = layers.Conv2D(base_filters * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional block 3
    x = layers.Conv2D(base_filters * 4, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten & Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="SimpleCNN")
    return model



def create_model(
    config: Config,
    model_type: str = "transfer",
    **kwargs
) -> tf.keras.Model:
    """
    Factory function to create a Keras model based on `model_type`.
    - model_type: "simple" or "transfer" or others you define
    - input_shape: tuple, the input shape
    - num_classes: number of classes for classification
    - **kwargs: additional parameters (e.g., trainable_base for transfer model)
    
    Example usage:
        model = create_model(
            model_type="transfer",
            input_shape=(224, 224, 3),
            num_classes=2,
            trainable_base=True
        )
    """
    my_config = Config()
    if model_type == "simple":
        model = build_simple_cnn(
            my_config.INPUT_SHAPE,
            base_filters=kwargs.get('base_filters', 32))
        
    elif model_type == "transfer":
        model  = build_pretrained_model(
            architecture= my_config.MODEL_ARCHITECTURE,
            input_shape= my_config.INPUT_SHAPE,
            trainable_base=kwargs.get('trainable_base', False),
            fine_tune_at=kwargs.get('fine_tune_at', None))
        
    elif model_type == "transfer_attention":
        model  = build_pretrained_attention_model(architecture= my_config.MODEL_ARCHITECTURE,
                            input_shape= my_config.INPUT_SHAPE,
                            trainable_base=kwargs.get('trainable_base', False),
                            fine_tune_at=kwargs.get('fine_tune_at', None))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model