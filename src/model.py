# model.py
import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout)
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16, InceptionV3, EfficientNetV2B0, EfficientNetV2B1, EfficientNetB0, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_v2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)
from src.config import Config

def custom_base_model_v1(input_shape, regulizer_value, dropout_value = None):
    """
    Builds the custom CNN base model for feature extraction.

    Includes input rescaling from [0, 255] to [-1, 1].

    Args:
        input_shape (tuple): The shape of the input images (e.g., (256, 256, 3)).

    Returns:
        tf.keras.models.Model: The Keras Model object representing the base feature extractor.
    """
    # --- Input and Preprocessing ---
    inputs = Input(shape=input_shape, name="input_image")
    # Scales input from [0, 255] to [-1, 1]
    # x = Rescaling(scale=1./127.5, offset=-1, name="rescaling")(inputs)

    # --- Feature Extractor Blocks ---

    # Block 1
    # Consider parameterizing filter counts if needed: filters_b1=32
    x = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv1_1")(inputs)
    x = BatchNormalization(name="bn1_1")(x)
    x = Activation('relu', name="relu1_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    # filters_b2=64
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
                name="conv2_1")(x)
    x = BatchNormalization(name="bn2_1")(x)
    x = Activation('relu', name="relu2_1")(x)
    # Optional: Add a second Conv layer
    # x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name="conv2_2")(x)
    # x = BatchNormalization(name="bn2_2")(x)
    # x = Activation('relu', name="relu2_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = Dropout(dropout_value, name="dropout1")(x)

    # Block 3
    # filters_b3=128
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv3_1")(x)
    x = BatchNormalization(name="bn3_1")(x)
    x = Activation('relu', name="relu3_1")(x)
    # Optional: Add a second Conv layer
    # x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name="conv3_2")(x)
    # x = BatchNormalization(name="bn3_2")(x)
    # x = Activation('relu', name="relu3_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = Dropout(dropout_value, name="dropout2")(x)

    # Block 4
    # filters_b4=256
    x = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv4_1")(x)
    x = BatchNormalization(name="bn4_1")(x)
    x = Activation('relu', name="relu4_1")(x)
    # Optional: Add a second Conv layer
    # x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name="conv4_2")(x)
    # x = BatchNormalization(name="bn4_2")(x)
    # x = Activation('relu', name="relu4_2")(x)
    base_model_output = MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    

    # --- End of Base Model ---
    # Output features before the classification head
    # Using GlobalAveragePooling2D as the standard output for a base model
    # You could also return the output of 'pool4' if you prefer a specific head structure
    # base_model_output = GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Create the base model instance
    base_model = Model(inputs=inputs, outputs=base_model_output, name="custom_cnn_base_v1")

    return base_model


def custom_base_model_v2(input_shape, regulizer_value, dropout_value = None):
    """
    Builds the custom CNN base model for feature extraction.

    Includes input rescaling from [0, 255] to [-1, 1].

    Args:
        input_shape (tuple): The shape of the input images (e.g., (256, 256, 3)).

    Returns:
        tf.keras.models.Model: The Keras Model object representing the base feature extractor.
    """
    # --- Input and Preprocessing ---
    inputs = Input(shape=input_shape, name="input_image")
    # Scales input from [0, 255] to [-1, 1]
    # x = Rescaling(scale=1./127.5, offset=-1, name="rescaling")(inputs)

    # --- Feature Extractor Blocks ---

    # Block 1
    # Consider parameterizing filter counts if needed: filters_b1=32
    x = Conv2D(32, kernel_size=(2, 2), strides = (1,1) , padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv1_1")(inputs)
    x = BatchNormalization(name="bn1_1")(x)
    x = Activation('relu', name="relu1_1")(x)
    x = Conv2D(32, kernel_size=(2, 2), strides = (1,1) , padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv1_2")(x)
    x = BatchNormalization(name="bn1_2")(x)
    x = Activation('relu', name="relu1_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = Dropout(0.2, name="dropout1")(x)

    # Block 2
    # filters_b2=64
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
                name="conv2_1")(x)
    x = BatchNormalization(name="bn2_1")(x)
    x = Activation('relu', name="relu2_1")(x)
    # Optional: Add a second Conv layer
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
                name="conv2_2")(x)
    x = BatchNormalization(name="bn2_2")(x)
    x = Activation('relu', name="relu2_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = Dropout(0.35, name="dropout2")(x)

    # Block 3
    # filters_b3=128
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv3_1")(x)
    x = BatchNormalization(name="bn3_1")(x)
    x = Activation('relu', name="relu3_1")(x)
    # Optional: Add a second Conv layer
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
               name="conv3_2")(x)
    x = BatchNormalization(name="bn3_2")(x)
    base_model_output = Activation('relu', name="relu3_2")(x)
    # base_model_output = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    # base_model_output = Dropout(0.4, name="dropout3")(x)

    # Block 4
    # filters_b4=256
    # x = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
    #            kernel_regularizer=tf.keras.regularizers.l2(regulizer_value),
    #            name="conv4_1")(x)
    # x = BatchNormalization(name="bn4_1")(x)
    # x = Activation('relu', name="relu4_1")(x)
    # # Optional: Add a second Conv layer
    # # x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name="conv4_2")(x)
    # # x = BatchNormalization(name="bn4_2")(x)
    # # x = Activation('relu', name="relu4_2")(x)
    # base_model_output = MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    

    # --- End of Base Model ---
    # Output features before the classification head
    # Using GlobalAveragePooling2D as the standard output for a base model
    # You could also return the output of 'pool4' if you prefer a specific head structure
    # base_model_output = GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Create the base model instance
    base_model = Model(inputs=inputs, outputs=base_model_output, name="custom_cnn_base_v2")

    return base_model

def build_pretrained_model( config: Config,
                            trainable_base: bool = False,
                            ):
    """
    architecture: str, one of {'resnet50','inceptionv3','efficientnetb0', ...}
    freeze: whether to freeze base model layers initially
    Returns a compiled model
    """
    architectures = ['ResNet50', 'MobileNetV2','InceptionV3', 'EfficientNetB0', 'EfficientNetV2B0', 'EfficientNetV2B1', 'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge']
    
    architecture = config.MODEL_ARCHITECTURE
    input_shape = config.INPUT_SHAPE
    head_dense_units = config.HEAD_DENSE_UNITS
    fine_tune_at = config.FINE_TUNE_FROM_LAYER
    
    if architecture.lower() == 'custom_v1':
        base_model = custom_base_model_v1(input_shape=input_shape,
                                          regulizer_value= config.L2_REGULARIZATION,
                                          dropout_value= config.DROPOUT_RATE)
        preprocess_func = lambda t: tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(t)
    elif architecture.lower() == 'custom_v2':
        base_model = custom_base_model_v2(input_shape=input_shape,
                                          regulizer_value= config.L2_REGULARIZATION,
                                          dropout_value= config.DROPOUT_RATE)
        preprocess_func = lambda t: tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(t)

    elif architecture.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
        preprocess_func = vgg16_preprocess

    elif architecture.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                                input_shape=input_shape)
        preprocess_func = resnet_preprocess
        
    elif architecture.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=input_shape)
        preprocess_func = mobilenetv2_preprocess

    elif architecture.lower() == 'efficientnetv2b1':
        base_model = EfficientNetV2B1(weights='imagenet', include_top=False,
                                input_shape=input_shape)
        preprocess_func = effnet_v2_preprocess
        
    elif architecture.lower() == 'convnexttiny':
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False,
                                input_shape=input_shape)
        preprocess_func = convnext_preprocess
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {architectures}")

    
    
    # # We freeze the base model in the first step to prevent weight deterioration
    # if not (trainable_base or ("custom" in architecture)):
    #     # Freeze entire base model for initial training
    #     base_model.trainable = False
    # else:
    #     if fine_tune_at is not None:
    #         # Freeze layers until fine_tune_at and unfreeze from that point onward
    #         for layer in base_model.layers[:fine_tune_at]:
    #             layer.trainable = False
    #         for layer in base_model.layers[fine_tune_at:]:
    #             layer.trainable = True
    #     else:
    #         base_model.trainable = True



    if not trainable_base and not ("custom" in architecture.lower()):
        # Freeze entire PRETRAINED base model for initial training
        base_model.trainable = False
        print("###############Base model frozen")
    else:
        # This block handles:
        # 1. Fine-tuning PRETRAINED models (trainable_base=True)
        # 2. Training CUSTOM model (trainable_base might be True or False, depends on use)
        if fine_tune_at is not None and not ("custom" in architecture.lower()):
            # Fine-tuning PRETRAINED from a specific layer
            # Set base_model itself to trainable first to allow inner layers to be set
            base_model.trainable = True # Important before setting individual layers!
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
            print(f"###############Fine-tuning from layer {fine_tune_at}")
            print(f"###############Number of trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}")
        else:
            # Handles CUSTOM model OR full fine-tuning of PRETRAINED model
            base_model.trainable = True
            print("############### Base model trainable")

    # Create an Input layer and add a Lambda layer for preprocessing.
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(preprocess_func, name="preprocessing_layer")(inputs)
    
    # Pass preprocessed inputs through the base model.
    features = base_model(x)


    if config.HEAD_ARCHITECTURE == "shallow":
        
        # Shallow Head (GAP -> Dense -> BN -> Dropout -> Output)
        x = layers.GlobalAveragePooling2D(name='head_gap')(features)
        x = layers.BatchNormalization(name='head_bn_1')(x)
        # single dense block
        x = layers.Dense(head_dense_units, activation='relu', 
                            name='head_dense_1',
                            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(x)
        x = layers.BatchNormalization(name='head_bn_2')(x)
        x = Activation('relu', name="head_relu_1")(x) 
        x = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_1')(x)
        # Final classification layer connected to the output of the second block 
        classification_output = layers.Dense(1, activation='sigmoid', 
                                        name='classification_output')(x)
    
    elif config.HEAD_ARCHITECTURE == "moderate":
    
        # Moderately Shallow Head (Two Dense Blocks)
        gap = layers.GlobalAveragePooling2D(name='head_gap')(features)
        bn1 = layers.BatchNormalization(name='head_bn_initial')(gap)     
        # First dense block
        dns1 = layers.Dense(head_dense_units, activation='relu', 
                            name='head_dense_1',
                            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(bn1)
        bn2 = layers.BatchNormalization(name='head_bn_1')(dns1)
        do1 = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_1')(bn2)
        # Second dense block (using head_dense_units//2)
        dns2 = layers.Dense(head_dense_units//2, activation='relu', 
                            name='head_dense_2',
                            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(do1)
        bn3 = layers.BatchNormalization(name='head_bn_2')(dns2)
        do2 = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_2')(bn3) # Last dropout
        # Final classification layer connected to the output of the second block
        classification_output = layers.Dense(1, activation='sigmoid', 
                                        name='classification_output')(do2)
    
    elif config.HEAD_ARCHITECTURE == "deep":
        # option 2: GlobalAveragePooling2D + BN + Dense256 + Dropout05 + Dense
        gap = layers.GlobalAveragePooling2D(name='head_gap')(features)
        bn1 = layers.BatchNormalization(name='head_bn_initial')(gap)  # helps with feature scale
        # First dense block
        dns1 = layers.Dense(head_dense_units, activation='relu', name = 'head_dense_1', kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(bn1)
        bn2 = layers.BatchNormalization(name='head_bn_1')(dns1)  # helps with feature scale
        do1 = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_1')(bn2)
        # Second dense block (using head_dense_units//2)
        dns2 = layers.Dense(head_dense_units//2, activation='relu', name = 'head_dense_2', kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(do1)
        bn3 = layers.BatchNormalization(name='head_bn_2')(dns2)  # helps with feature scale
        do2 = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_2')(bn3)
        # Second dense block (using head_dense_units//4)
        dns3 = layers.Dense(head_dense_units//4, activation='relu', name = 'head_dense_3', kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION))(do2)
        do3 = layers.Dropout(config.DROPOUT_RATE, name='head_dropout_3')(dns3)
        # Final classification layer connected to the output of the second block
        classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(do3)
    

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
    if model_type == "simple":
        model = build_simple_cnn(
            config.INPUT_SHAPE,
            base_filters=kwargs.get('base_filters', 32))
        
    elif model_type == "transfer":
        model  = build_pretrained_model(
            config = config,
            trainable_base=kwargs.get('trainable_base', False),
            )
        
    elif model_type == "transfer_attention":
        model  = build_pretrained_attention_model(architecture= config.MODEL_ARCHITECTURE,
                            input_shape= config.INPUT_SHAPE,
                            trainable_base=kwargs.get('trainable_base', False),
                            fine_tune_at=kwargs.get('fine_tune_at', None),)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model