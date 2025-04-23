import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Config:
    """
        A configuration class that contains the main parameters for the project.
        
        Attributes:
        IMAGE_DIR (str): Directory for processed images.
        ANNOTATION_FILE (str): Path to the annotation file.
        OUTPUT_DIR (str): Output directory.
        DATA_DIR (str): Base data directory.
        TARGET_DIM (int): The target dimension (in pixels) for square images.
        MASK_VALUE (int): The constant value used for padding background regions.
        ANTIALIZING_IN_RESIZING (bool): Whether to use antialiasing in resizing.
        TRAIN_RATIO (float): Ratio of images used for training.
        VAL_RATIO (float): Ratio of images used for validation.
        TEST_RATIO (float): Ratio of images used for testing.
        BATCH_SIZE (int): Batch size used in model training.
        RANDOM_SEED (int): Random seed for reproducibility.
        AUGMENT_DATA (bool): Flag to indicate whether to augment data.
        SHUFFLE_DATASET (bool): Flag to indicate whether to shuffle dataset.
        NORMALIZE_IMAGES (bool): Flag to indicate if images are normalized.
        MASK_BG (bool): Flag to indicate if background should be masked.
        REMOVE_DARK_IMAGES (bool): Flag to indicate if dark images should be removed.
        TARGET_CLASS (str): The target class for classification tasks.
        MODEL_ARCHITECTURE (str): Name of the model architecture.
        L2_REGULARIZATION (float): L2 regularization weight.
        DROPOUT_RATE (float): Dropout rate.
        INITIAL_LR (float): Initial learning rate.
        NUM_INITIAL_EPOCHS (int): Number of initial epochs for training.
        NUM_FINE_TUNE_EPOCHS (int): Number of fine-tuning epochs for training.
        FINE_TUNE_LR (float): Learning rate for fine-tuning.
        FINE_TUNE_FROM_LAYER (int): Layer index from which to unfreeze for fine-tuning.
        DARK_IMAGE_THRESHOLD (float): Threshold for determining dark images.
        POLYGON_SMOOTHING_TOLERANCE (float): Tolerance for smoothing tooth polygons.

        IMAGE_PVALUE_TYPE: np.dtype  # Image pixel value type (e.g., np.float32)
        RESCALE_PIXELS: list  # Rescale pixel values for example to [-1, 1]
        OVERSAMPLE_FACTOR: int # Oversampling factor for minority classes. 
                                ((OVERSAMPLE_FACTOR-1) * num_minority_samples ) number of augmentations from the target class will be added to the dataset.
                                Because we have considered the target class to be the minority class in the master dataset.
                                The factor can be the majority/minority ratio.
    """

    TARGET_CLASS: str = "Pla"
    IMAGE_DIR: str = "data/raw"
    RAW_DS_DIR: str = "data/raw/complete_toothwise_annotations.json"
    ANNOTATION_FILE: str = "data/processed"#"data/interim"
    PROCESSED_DIR: str = "data/processed"
    OUTPUT_DIR: str = "data/processed"
    DATA_DIR: str = "/home/arash/tooth_classification_project"#"C:/Users/anedaeij23/Project/tooth_classification_project"#
    LOG_DIR: str = "logs"
    TRAIN_JSON_NAME: str = f"{TARGET_CLASS}_filtered_train.json"
    VAL_JSON_NAME: str = f"{TARGET_CLASS}_filtered_val.json"
    TEST_JSON_NAME: str = f"{TARGET_CLASS}_filtered_test.json"
    TOOTH_NUMBER_LIST: list = field(default_factory=lambda:[11, 12, 13, 14, 15, 16,# 17, 18,
                                                            21, 22, 23, 24, 25, 26,# 27, 28,
                                                            31, 32, 33, 34, 35, 36,# 37, 38,
                                                            41, 42, 43, 44, 45, 46,# 47, 48,
                                                            # 51, 52, 53, 54, 55, 56, 57, 58,
                                                            # 61, 62, 63, 64, 65, 66, 67, 68,
                                                            # 71, 72, 73, 74, 75, 76, 77, 78,
                                                            # 81, 82, 83, 84, 85, 86, 87, 88,
                                                            ])
    
    TARGET_DIM: int = 256
    INPUT_SHAPE: tuple = (TARGET_DIM, TARGET_DIM, 3)
    MASK_VALUE: int = 20 #if set to (5-49), the background will be blur and the value will be used for skimage.filter.gaussian sigma value
    ANTIALIZING_IN_RESIZING: bool = False
    
    
    TRAIN_VAL_TEST_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    BATCH_SIZE: int = 32
    RANDOM_SEED: int = 42
    AUGMENT_DATA: bool = True
    NON_AUG_AUG_PERCENT: float = 0.3
    SHUFFLE_DATASET: bool = True
    NORMALIZE_IMAGES: bool = True
    MASK_BG: bool = True
    REMOVE_DARK_IMAGES: bool = True
    
    DARK_IMAGE_THRESHOLD: float = 0.25
    BLUR_IMAGE_THRESHOLD: float = 0.5 # images lower than this threshold will be removed from the dataset. disabled if 0
    MASK_POLYGON_SMOOTHING: bool = False
    POLYGON_SMOOTHING_TOLERANCE: float = 0.015
    IMAGE_PVALUE_TYPE: np.dtype = np.float32 #np.float32
    RESCALE_PIXELS: tuple = (0, 255) #define based on model requirement. resnet 
    OVERSAMPLE_FACTOR: int = 2 
    
    
    MODEL_ARCHITECTURE: str = "resnet50"# custome_v1, resnet50, ...
    TAP_INTO_BASE_MODEL: str = False
    HEAD_DENSE_UNITS: int = 256
    HEAD_ARCHITECTURE: str = "moderate_combo"#"shallow", "moderate", "deep"
    L2_REGULARIZATION: float = 0.003# reduced from 0.001
    DROPOUT_RATE: float = 0.25# reduced from 0.3
    INITIAL_LR: float = 0.04
    NUM_INITIAL_EPOCHS: int = 2
    NUM_FINE_TUNE_EPOCHS: int = 0
    FINE_TUNE_LR: float = 0.001
    FINE_TUNE_FROM_LAYER: int = 0 #conv4 143#conv5
    
    OPTIMIZER: str = 'SGD' # 'Adam', 'SGD', 'RMSprop'
    MOMENTUM: float = 0.85 #applicable if SGD or RMSprop is selected
    LOSS_FUNC: str = 'BinaryCrossentropy'
    BFC_GAMMA: float = 2.0 # applicable if BinaryFocalCrossentropy is selected
    CLASS_WEIGHTS: dict = field(default_factory=lambda:{0: 0.5, 1:1.5})
    METRIC_THRESHOLDS: list = field(default_factory=lambda:[0.5]) # Threshold for threshold dependant metrics calculation (can be a list of thresholds)

    
    # Follwoing list of dictionaries define the experiment space. these paramateras are a subset of the parameters above that can be set to define the space.
    EXPERIMENT_PARAMS: list = field(default_factory = lambda:
                                [
    # --- BLOCK 0 --- base model search_in Exp2 changed lr, do, 
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 16, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 277, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 165, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

##########
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 16, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 277, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 155, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 165, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 15},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 20, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    
####





    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0},


    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'mobilenetv2', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'efficientnetv2b1', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'convnexttiny', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0},

    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0},
    # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0},


    # --- Block 1: Baseline & Effect of Tapping Point (Frozen Base) ---
    # # Baseline: Default InceptionV3 output, frozen base, standard head
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # # Compare: Use mixed7 output, frozen base, standard head
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False (Base here is up to mixed7)

    # # --- Block 2: Effect of Fine-Tuning Strategy (Using Default Output) ---
    # # Compare Baseline (Frozen) vs. Fine-tuning later layers vs. Fine-tuning more layers
    # # Baseline Ref: {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': None, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": False}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 249, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 25}, # trainable_base=True, tune from ~mixed7/8
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 172, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": False, "NUM_FINE_TUNE_EPOCHS": 25}, # trainable_base=True, tune from ~mixed4

    # # --- Block 3: Effect of Fine-Tuning Strategy (Using mixed7 Output) ---
    # # Compare Frozen vs. Fine-tuning layers *before* mixed7
    # # Baseline Ref: {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': None, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 172, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 25}, # trainable_base=True, tune from ~mixed4 up to mixed7

    # # --- Block 4: Effect of Head Capacity (Using mixed7 Output, Frozen Base) ---
    # # Baseline Ref: {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': None, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 512, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False

    # # --- Block 5: Effect of Head Architecture (Using mixed7 Output, Frozen Base, 256 Units) ---
    # # Baseline Ref: {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': None, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "moderate", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "deep_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
    # # --- Block 6: Effect of Background Masking (Using mixed7 Output, Frozen Base, 256 Units, Shallow Head) ---
    # # Baseline Ref: {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': None, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True}, # trainable_base=False
    # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'inceptionv3', 'FINE_TUNE_FROM_LAYER': 0, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": True, "NUM_FINE_TUNE_EPOCHS": 0}, # trainable_base=False
                                    
                                    # [
    #     {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 15, 'HEAD_ARCHITECTURE': "shallow_combo", "TAP_INTO_BASE_MODEL": False},
    #     {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 15, 'HEAD_ARCHITECTURE': "deep"},
    #     {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 15, 'HEAD_ARCHITECTURE': "deep_combo"},
    #     {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'vgg16', 'FINE_TUNE_FROM_LAYER': 15, 'HEAD_ARCHITECTURE': "deep_combo"},


        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 81, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"}#"moderate"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 81, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 20, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 6, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'custom_v1', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},


        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 81, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 81, 'HEAD_ARCHITECTURE': "moderate"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 20, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 6, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'resnet50', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},

        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 110, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 20, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 6, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'EfficientNetV2B1', 'FINE_TUNE_FROM_LAYER': 198, 'HEAD_ARCHITECTURE': "shallow"},

        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 53, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 20, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 6, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'ConvNeXtTiny', 'FINE_TUNE_FROM_LAYER': 126, 'HEAD_ARCHITECTURE': "shallow"},
        
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 134, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 128, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 0, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True, "MASK_BG": False, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 20, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 6, "AUGMENT_DATA": True,  "NORMALIZE_IMAGES": True,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 128, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
        # {"MASK_VALUE": 10, "AUGMENT_DATA": True, "NORMALIZE_IMAGES": False,  "MASK_BG": True, "HEAD_DENSE_UNITS": 256, "MODEL_ARCHITECTURE": 'MobileNetV2', 'FINE_TUNE_FROM_LAYER': 143, 'HEAD_ARCHITECTURE': "shallow"},
    ])
    
    
    HP_SPACE = {
    'head_dense_units': {
        'min_value': 128,
        'max_value': 512,
        'step': 128,
        'type': 'int'
    },
    'dropout_rate': {
        'min_value': 0.1,
        'max_value': 0.5,
        'step': 0.1,
        'type': 'float'
    },
    'l2_regularization': {
        'min_value': 1e-4,
        'max_value': 1e-1,
        'sampling': 'log',
        'type': 'float'
    },
    'initial_lr': {
        'min_value': 1e-3,
        'max_value': 1e-1,
        'sampling': 'log',
        'type': 'float'
    },
    # 'fine_tune_lr': {
    #     'min_value': 1e-5,
    #     'max_value': 1e-3,
    #     'sampling': 'log',
    #     'type': 'float'
    # },
    'class_weight_neg': {  # weight for class 0
        'min_value': 0.2,
        'max_value': 1.0,
        'step': 0.2,
        'type': 'float'
    },
    'class_weight_pos': {  # weight for class 1
        'min_value': 1.0,
        'max_value': 2.0,
        'step': 0.1,
        'type': 'float'
    }
}