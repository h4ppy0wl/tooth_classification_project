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
    
    IMAGE_DIR = "data/processed"
    ANNOTATION_FILE = "data/annotations.json"
    OUTPUT_DIR = "data/processed"
    DATA_DIR = "C:/Users/anedaeij23/Project/tooth_classification_project"
    

    
    
    TARGET_DIM = 256
    MASK_VALUE = 0#128
    ANTIALIZING_IN_RESIZING = True

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    BATCH_SIZE = 32
    RANDOM_SEED = 42
    AUGMENT_DATA = False
    SHUFFLE_DATASET = True
    NORMALIZE_IMAGES = False
    MASK_BG = True
    REMOVE_DARK_IMAGES = True
    

    TARGET_CLASS = "Pla"
    
    MODEL_ARCHITECTURE = "resnet50"
    L2_REGULARIZATION = 0.001
    DROPOUT_RATE = 0.5
    INITIAL_LR = 1e-3
    NUM_INITIAL_EPOCHS = 10
    NUM_FINE_TUNE_EPOCHS = 10
    FINE_TUNE_LR = 1e-4
    FINE_TUNE_FROM_LAYER = 22
    
    # additional parameters that you probably don't need to change
    DARK_IMAGE_THRESHOLD = 0.25
    POLYGON_SMOOTHING_TOLERANCE = 0.015
    

    # Add additional parameters as needed.