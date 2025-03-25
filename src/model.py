# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from src.config import Config
from tensorflow.keras.applications.resnet

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

    elif architecture.lower() == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=input_shape)
        preprocess_func = inception_preprocess

    elif architecture.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=input_shape)
        preprocess_func = effnet_preprocess

    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {architectures}")

    
    
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
    config,
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
            base_filters=kwargs.get('base_filters', 32)
        )
    elif model_type == "transfer":
        model = build_pretrained_model(
            architecture= my_config.MODEL_ARCHITECTURE,
            input_shape= my_config.INPUT_SHAPE,
            trainable_base=kwargs.get('trainable_base', False),
            fine_tune_at=kwargs.get('fine_tune_at', None)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model





#check later:

import tensorflow as tf
import numpy as np
from skimage import io
import os
from src.config import Config

# Assume these functions are imported from your module:
# dental_gray_world_white_balance, mask_background, pad_and_resize

def tf_preprocess_image(image_path):
    """
    Wraps your existing Python preprocessing logic for use in a tf.data pipeline.
    This function reads, decodes, and applies your processing (white balance, optional masking,
    and pad/resize) to a given image path.
    """
    def _preprocess(path):
        # path is a numpy string
        path_str = path.decode("utf-8")
        # Load the image
        if not os.path.exists(path_str):
            # Return a dummy image if not exists.
            return np.zeros(Config.INPUT_SHAPE, dtype=np.uint8)
        image = io.imread(path_str)
        if image is None:
            return np.zeros(Config.INPUT_SHAPE, dtype=np.uint8)
        # Optionally normalize/white balance        
        if Config.NORMALIZE_IMAGES:
            image = dental_gray_world_white_balance(image)
        # Optionally mask background; if required you need the polygon
        # Here, you may load a polygon or pass it in via additional logic.
        # For this example, we'll assume masking is desired.
        # You must modify this if you have the polygon info attached to each filepath.
        if Config.MASK_BG:
            # Dummy polygon for example; replace with actual polygon if available.
            dummy_poly = {"all_points_x": [0, image.shape[1]], "all_points_y": [0, image.shape[0]]}
            image = mask_background(image, dummy_poly)
        # Pad and resize image
        image = pad_and_resize(image, target_dim=Config.TARGET_DIM, mask_value=Config.MASK_VALUE)
        return image.astype(np.float32)  # Ensure data type compatibility

    # Wrap the _preprocess function into a tf.py_function.
    image = tf.py_function(func=_preprocess, inp=[image_path], Tout=tf.float32)
    # Set static shape so that downstream ops know the dimensions.
    image.set_shape(Config.INPUT_SHAPE)
    return image


def build_tf_dataset(image_paths, labels, batch_size=Config.BATCH_SIZE):
    """
    Builds a TensorFlow dataset that applies the preprocessing graph.
    Args:
        image_paths: List of image file paths.
        labels: List of labels.
    Returns:
        A tf.data.Dataset instance.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    def _load_and_preprocess(path, label):
        image = tf_preprocess_image(path)
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds