import numpy as np
from dataclasses import dataclass

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
    ANNOTATION_FILE: str = "data/interim"
    OUTPUT_DIR: str = "data/processed"
    DATA_DIR: str = "C:/Users/anedaeij23/Project/tooth_classification_project"
    TRAIN_JSON_NAME: str = f"{TARGET_CLASS}_filtered_train.json"
    VAL_JSON_NAME: str = f"{TARGET_CLASS}_filtered_val.json"
    TEST_JSON_NAME: str = f"{TARGET_CLASS}_filtered_test.json"
    
    TARGET_DIM: int = 256
    INPUT_SHAPE: tuple = (TARGET_DIM, TARGET_DIM, 3)
    MASK_VALUE: int = 0
    ANTIALIZING_IN_RESIZING: bool = False
    
    
    TRAIN_VAL_TEST_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    BATCH_SIZE: int = 32
    RANDOM_SEED: int = 42
    AUGMENT_DATA: bool = True
    SHUFFLE_DATASET: bool = True
    NORMALIZE_IMAGES: bool = True
    MASK_BG: bool = True
    REMOVE_DARK_IMAGES: bool = True
    
    DARK_IMAGE_THRESHOLD: float = 0.25
    MASK_POLYGON_SMOOTHING: bool = False
    POLYGON_SMOOTHING_TOLERANCE: float = 0.015
    IMAGE_PVALUE_TYPE: np.dtype = np.float32
    RESCALE_PIXELS: tuple = (0, 255) #define based on model requirement. resnet 
    OVERSAMPLE_FACTOR: int = 3 
    
    
    MODEL_ARCHITECTURE: str = "resnet50"
    L2_REGULARIZATION: float = 0.001
    DROPOUT_RATE: float = 0.5
    INITIAL_LR: float = 1e-1
    NUM_INITIAL_EPOCHS: int = 2
    NUM_FINE_TUNE_EPOCHS: int = 2
    FINE_TUNE_LR: float = 1e-4
    FINE_TUNE_FROM_LAYER: int = 22
    

    
    