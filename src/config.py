class Config:
    """
    A configuration class that contains the main parameters for the project.
    
    Attributes:
        TARGET_DIM (int): The target dimension (in pixels) for square images after padding and resizing.
        MASK_VALUE (int): The constant value used for padding background regions.
        TRAIN_RATIO (float): Ratio of images used for training.
        VAL_RATIO (float): Ratio of images used for validation.
        TEST_RATIO (float): Ratio of images used for testing.
        BATCH_SIZE (int): Batch size used in model training.
        RANDOM_SEED (int): Random seed for reproducibility.
        TARGET_CLASS (str): The target class (e.g., "plaque") for classification tasks.
    """
    
    IMAGE_DIR = "data/images"
    ANNOTATION_FILE = "data/annotations.json"
    OUTPUT_DIR = "data/processed"
    
    MASK_BG = True
    
    
    TARGET_DIM = 224
    MASK_VALUE = 128

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    BATCH_SIZE = 32
    RANDOM_SEED = 42

    TARGET_CLASS = "Pla"
    
    
    # additional parameters that you probably don't need to change
    DARK_IMAGE_THRESHOLD = 0.25
    POLYGON_SMOOTHING_TOLERANCE = 0.015
    

    # Add additional parameters as needed.