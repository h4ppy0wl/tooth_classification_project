# model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_simple_cnn(
    input_shape=(224, 224, 3), 
    num_classes=2, 
    base_filters=32
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
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name="SimpleCNN")
    return model


def build_transfer_model(
    input_shape=(224, 224, 3),
    num_classes=2,
    trainable_base: bool = False
) -> tf.keras.Model:
    """
    Build a transfer learning model using ResNet50 as the base.
    - input_shape: The shape of input images
    - num_classes: Number of output classes
    - trainable_base: If True, the base ResNet layers are trainable; 
                      if False, they are frozen.
    """
    # Pretrained ResNet50 (ImageNet weights)
    base_model = ResNet50(
        include_top=False,  # Exclude the final FC layer
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze or unfreeze base model layers
    base_model.trainable = trainable_base
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Optional: Preprocessing for ResNet (if needed)
    # e.g., x = tf.keras.applications.resnet50.preprocess_input(inputs)
    # But you can skip if data is already scaled or you handle it in your pipeline
    x = inputs
    
    x = base_model(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name="Transfer_ResNet50")
    return model


def create_model(
    model_type: str = "simple",
    input_shape=(224, 224, 3),
    num_classes=2,
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
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=kwargs.get('base_filters', 32)
        )
    elif model_type == "transfer":
        model = build_transfer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            trainable_base=kwargs.get('trainable_base', False)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compile the model
    # You can also move compilation outside, but here is a quick example
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 1e-3)),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
