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
        OVERSAMPLE_FACTOR: int # Oversampling factor for minority classes. ((OVERSAMPLE_FACTOR-1) * num_minority_samples ) augmentations will be added to the dataset
    """

    IMAGE_DIR: str = "data/processed"
    ANNOTATION_FILE: str = "data/annotations.json"
    OUTPUT_DIR: str = "data/processed"
    DATA_DIR: str = "C:/Users/anedaeij23/Project/tooth_classification_project"
    
    TARGET_DIM: int = 256
    INPUT_SHAPE: tuple = (TARGET_DIM, TARGET_DIM, 3)
    MASK_VALUE: int = 0
    ANTIALIZING_IN_RESIZING: bool = True
    
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    BATCH_SIZE: int = 32
    RANDOM_SEED: int = 42
    AUGMENT_DATA: bool = False
    SHUFFLE_DATASET: bool = True
    NORMALIZE_IMAGES: bool = False
    MASK_BG: bool = True
    REMOVE_DARK_IMAGES: bool = True
    
    TARGET_CLASS: str = "Pla"
    MODEL_ARCHITECTURE: str = "resnet50"
    L2_REGULARIZATION: float = 0.001
    DROPOUT_RATE: float = 0.5
    INITIAL_LR: float = 1e-3
    NUM_INITIAL_EPOCHS: int = 10
    NUM_FINE_TUNE_EPOCHS: int = 10
    FINE_TUNE_LR: float = 1e-4
    FINE_TUNE_FROM_LAYER: int = 22
    
    DARK_IMAGE_THRESHOLD: float = 0.25
    MASK_POLYGON_SMOOTHING: bool = False
    POLYGON_SMOOTHING_TOLERANCE: float = 0.015 
    IMAGE_PVALUE_TYPE: np.dtype = np.float32 
    RESCALE_PIXELS: list = [0, 255]
    OVERSAMPLE_FACTOR: int = 4 
    
    