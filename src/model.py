# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from src.config import Config

def build_pretrained_model(architecture = Config.MODEL_ARCHITECTURE, input_shape=(Config.TARGET_DIM,Config.TARGET_DIM,3),
                            trainable_base: bool = False,
                            fine_tune_at: int = None):
    """
    architecture: str, one of {'resnet50','inceptionv3','efficientnetb0', ...}
    freeze: whether to freeze base model layers initially
    Returns a compiled model
    """

    if architecture.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)
        preprocess_func = resnet_preprocess

    elif architecture.lower() == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=input_shape)
        preprocess_func = inception_preprocess

    elif architecture.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=input_shape)
        preprocess_func = effnet_preprocess

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    
    
    # Optionally freeze
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

    # Add a custom classification head
    # option 1: GlobalAveragePooling2D + Dense128 + Dropout02 + Dense
    # x = layers.GlobalAveragePooling2D()(base_model.output)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(Config.DROPOUT_RATE)(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    
    # option 2: GlobalAveragePooling2D + BN + Dense256 + Dropout05 + Dense
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)  # helps with feature scale
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # option 3: Flatten + Dense256 + Dropout05 + Dense
    # x = layers.Flatten()(x)  # Flatten the spatial feature maps
    # x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(x)
    # x = layers.Dropout(Config.DROPOUT_RATE)(x)
    # output = layers.Dense(1, activation='sigmoid')(x)
    

    model = Model(inputs=base_model.input, outputs=output, name=f'{architecture}_model')



    return model, preprocess_func


def build_simple_cnn(
    input_shape= (Config.TARGET_DIM,Config.TARGET_DIM,3), 
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
    if model_type == "simple":
        model = build_simple_cnn(
            base_filters=kwargs.get('base_filters', 32)
        )
    elif model_type == "transfer":
        model = build_pretrained_model(
            trainable_base=kwargs.get('trainable_base', False),
            fine_tune_at=kwargs.get('fine_tune_at', None)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
