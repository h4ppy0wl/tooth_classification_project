# data_pipeline.py

import os
import glob
from typing import List, Tuple, Dict
import numpy as np
import cv2 
import skimage
from skimage import io, transform, draw, color, measure
import tensorflow as tf
import random
import json
from src.config import Config
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# import pandas as pd

class InputStream:
    def __init__(self, data):
            self.data = data
            self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


    @staticmethod
    def access_bit(data1, num):
        """ from bytes array to bits by num position"""
        base = int(num // 8)
        shift = 7 - int(num % 8)
        return (data1[base] & (1 << shift)) >> shift


    @staticmethod
    def bytes2bit(data):
        """ get bit string from bytes data"""
        string1 = ''.join([str(InputStream.access_bit(data, i)) for i in range(len(data) * 8)])
        return string1


    @staticmethod
    def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
        """
        Converts rle to image mask
        Args:
            rle: your long rle
            height: original_height
            width: original_width

        Returns: np.array
        """

        rle_input = InputStream(InputStream.bytes2bit(rle))

        num = rle_input.read(32)
        word_size = rle_input.read(5) + 1
        rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
        # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

        i = 0
        out = np.zeros(num, dtype=np.uint8)
        while i < num:
            x = rle_input.read(1)
            j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
            if x:
                val = rle_input.read(word_size)
                out[i:j] = val
                i = j
            else:
                while i < j:
                    val = rle_input.read(word_size)
                    out[i] = val
                    i += 1

        image = np.reshape(out, [height, width, 4])[:, :, 3]
        return image
    
    @staticmethod
    def polygon_to_mask(polygon: Dict[str, List[int]], height: int, width: int) -> np.array:
        """
        Converts polygon data to a binary mask.
        Args:
            polygon: Dictionary with 'all_points_x' and 'all_points_y' lists.
            height: Height of the output mask.
            width: Width of the output mask.

        Returns: np.array
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array([polygon['all_points_x'], polygon['all_points_y']]).T
        rr, cc = skimage.draw.polygon(points[:, 1], points[:, 0], mask.shape)
        mask[rr, cc] = 1
        return mask

def mask_to_label_studio_rle(mask):
    """
    Converts a 2D binary mask to Label Studio RLE format, starting with zero.
    
    Args:
        mask (np.ndarray): 2D binary mask (0 for background, 1 for object).
        
    Returns:
        list: RLE list in Label Studio format, starting with zero.
    """
    # Flatten the mask to a 1D array
    flat_mask = mask.flatten()
    
    # Initialize RLE with an initial zero
    rle = [0]
    
    # Track the current pixel value and the run length
    current_value = flat_mask[0]
    count = 0
    
    # Iterate through the flattened mask
    for pixel in flat_mask:
        if pixel == current_value:
            count += 1
        else:
            # Append the count to RLE when the value changes
            rle.append(int(count))
            # Reset count and update to the new pixel value
            current_value = pixel
            count = 1
    
    # Append the final run length
    rle.append(int(count))
    
    return rle

def mask_to_polygon(mask):
    """
    Converts a binary mask to a polygon.

    Args:
        mask (numpy.ndarray): The binary mask image.

    Returns:
        tuple: A tuple containing two lists: `all_points_x` and `all_points_y` representing the polygon's vertices.
    """
    mask = mask.astype(np.uint8)*255
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours were found, return empty lists
    if not contours:
        return [], []

    # Extract the first contour (assuming only one object in the mask)
    contour = contours[0]

    # Flatten the contour to a 1D array
    contour = contour.reshape(-1, 2)

    # Separate x and y coordinates
    all_points_x = [int(x) for x in contour[:, 0]]
    all_points_y = [int(y) for y in contour[:, 1]]

    return all_points_x, all_points_y

def convert_int32_to_int(obj):
  """
  Converts int32 values in a dictionary to regular integers for JSON serialization.
  """
  if isinstance(obj, dict):
    return {convert_int32_to_int(k): convert_int32_to_int(v) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [convert_int32_to_int(i) for i in obj]
  else:
    return obj if not isinstance(obj, np.int32) else int(obj)  # convert int32 to int

def dental_gray_world_white_balance(image_rgb):
    """
    A preprocessing function to apply modified gray-world white balance for dental images.
    Preserves red (gums/tongue) while balancing white (teeth).
    """
    img_float = image_rgb.copy()
    # Convert to float for processing
    if image_rgb.dtype == np.uint8:
        img_float = image_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert to HSV to detect white (teeth) and red (gums/tongue)
    img_hsv = color.rgb2hsv(img_float)

    # Create a "teeth mask" (high brightness)
    # teeth_mask = img_hsv[..., 2] > 0.75  # V (brightness) threshold for teeth

    # Create a "red mask" (gums/tongue)
    # red_mask = ((img_hsv[..., 0] > 0.95) | (img_hsv[..., 0] < 0.05)) & (img_hsv[..., 1] > 0.4)  # H (hue) for red
    blue_mask = ((img_hsv[..., 0] > 0.43) & (img_hsv[..., 0] < 0.70)) & ((img_hsv[..., 1] > 0.30) &((img_hsv[..., 1] < 0.55))) #this is glove mask
    black_mask = ((img_hsv[..., 0] > 0.95) | (img_hsv[..., 0] < 0.05)) & (img_hsv[..., 1] > 0.55)  # empty spaces in the images that are dark and represent the empty space inside the mouse

    # Compute mean values of each channel
    avg_v = np.mean(img_hsv[...,2][~black_mask])#[~np.logical_or(blue_mask, black_mask)])
    avg_r = np.mean(img_float[:, :, 0][~np.logical_or(blue_mask, black_mask)])#[~red_mask])  # Avoid red region
    avg_g = np.mean(img_float[:, :, 1][~np.logical_or(blue_mask, black_mask)])  # Keep green as reference
    avg_b = np.mean(img_float[:, :, 2][~np.logical_or(blue_mask, black_mask)])#[~teeth_mask])  # Avoid white (teeth) region

    # Compute global gray mean
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    factor_scaler = 2

    factor = 0.8 + factor_scaler*((avg_v-1)**2)
    # print(filename)
    # print(avg_v)
    # print(factor)
    # Scale each channel (avoid correcting teeth & red too much)
    img_float[:, :, 0] *= (factor+0.2)*(avg_gray / avg_r)  # Red correction (skip red_mask)
    img_float[:, :, 1] *= factor*(avg_gray / avg_g)  # Green correction (normal)
    img_float[:, :, 2] *= factor*(avg_gray / avg_b)  # Blue correction (skip teeth_mask)

    # Clip values to [0,1] to avoid artifacts
    img_float = np.clip(img_float, 0, 1)

    return img_float#skimage.img_as_ubyte(img_float)  # Convert back to uint8

def is_darker_than_threshold(image_path, threshold: float= Config.DARK_IMAGE_THRESHOLD):
    """
    Determines if the mean intensity of an image is darker than a given threshold.

    Parameters:
    image_path (str): The file path to the image.
    threshold (float, optional): The intensity threshold to compare against. Default is 0.25.

    Returns:
    bool: True if the mean intensity of the image is less than the threshold, False otherwise.
    """
    image = io.imread(image_path)
    image_gray = color.rgb2gray(image)  # Convert to grayscale
    if image_gray.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    mean_intensity = np.mean(image_gray)
    
    return mean_intensity < threshold

def remove_dark_images_from_json(json_data: dict, image_base_path: str) -> dict:
    """
    Iterates over each entry in the JSON data, and for each tooth in teeth_data,
    removes those entries for which is_darker_than_threshold() returns True (i.e. tooth images considered too dark).

    If an entry's teeth_data becomes empty after filtering, the entire entry is removed.
    
    Args:
        json_data: Dictionary representing the JSON structure.
        image_base_path: Base folder path where the tooth images are stored.
        
    Returns:
        Filtered JSON data with dark tooth images removed.
    """
    filtered_data = {}
    for key, entry in json_data.items():
        teeth_data = entry["teeth_data"]
        filtered_teeth = {}
        for tooth_key, tooth_entry in teeth_data.items():
            tooth_image_filename = tooth_entry["tooth_image_filename"]
            # print(f"Checking image: {tooth_image_filename}")
            if not tooth_image_filename:
                print("Missing image filename for tooth")
                continue
            full_image_path = os.path.join(image_base_path, tooth_image_filename)
            # print(f"Checking image: {tooth_image_filename}")
            if is_darker_than_threshold(full_image_path):  # Skip dark images
                print(f"{tooth_image_filename} was dark, removed")
                continue
            filtered_teeth[tooth_key] = tooth_entry
        # Only add entry if at least one tooth passed our brightness check
        if len(filtered_teeth) > 0:
            entry["teeth_data"] = filtered_teeth
            filtered_data[key] = entry
    return filtered_data

def parse_dataset_json(json_path: str, config: Config) -> list:
    """
    Reads a JSON file containing a list of samples and returns an oversampled list of records.
    
    Each record is a dictionary with fields:
        - 'image_filename': str, path to the image file
        - 'mask_polygon': any shape/format you store the polygon in
        - 'label': int (e.g., 1 for 'with plaque', 0 for 'without plaque')
        - 'augmentation_label': bool (True if this record should be augmented)
    
    The oversampling rate for 'with plaque' examples is defined by config.oversample_factor.
    """
    # 1) Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)  # Expecting a list of dicts

    # 2) Build a list of base records (augmentation_label=False by default)
    records = []
    for item in data:  # each item is one full image
        image_path = item['tooth_image_path']
        for tooth in item['teeth']:
            r = {
                'image_path': image_path,
                'bbox': tooth['bbox'],
                'label': tooth['label'],
                'augmentation_label': False
            }
            records.append(r)
            
    if config.OVERSAMPLE_FACTOR == 1:
        random.shuffle(records)
        return records
    # 3) Separate "with plaque" (label==1) from "without plaque" (label==0)
    with_plaque_records = [r for r in records if r['label'] == config.TARGET_CLASS]
    # Optionally, you can also do: without_plaque_records = [r for r in records if r['label'] != config.TARGET_CLASS]

    # 4) Create extra copies of the minority (with plaque) records 
    #    and set augmentation_label=True for them
    oversampled_records = []
    # E.g., if oversample_factor=3, each with-plaque record is repeated 2 additional times
    for r in with_plaque_records:
        for _ in range(config.OVERSAMPLE_FACTOR - 1):
            new_record = r.copy()
            new_record['augmentation_label'] = True
            oversampled_records.append(new_record)

    # 5) Merge original + extra augmented records
    all_records = records + oversampled_records

    # 6) Shuffle so the augmented records are interspersed randomly
    random.shuffle(all_records)

    return all_records

def smooth_polygon(polygon: dict, config: Config) -> dict:
    """
    Smooths the jagged borders of a polygon by approximating its shape with fewer vertices.

    This function receives a polygon defined by its 'all_points_x' and 'all_points_y' keys and uses 
    skimage.measure.approximate_polygon to reduce the stepped edges caused by software-generated annotation.
    The tolerance parameter controls the degree of smoothing (a higher value results in a smoother polygon).

    Parameters:
        polygon (dict): A dictionary containing two keys, "all_points_x" and "all_points_y", each mapping to a list of 
                        integers defining the polygon vertices.
        tolerance (float): The maximum distance from the original polygon points to the approximated points.
                           Higher values yield a smoother polygon shape.

    Returns:
        dict: A new polygon dictionary with smoothed "all_points_x" and "all_points_y" values.
    """
    # Convert polygon points to a N x 2 numpy array.
    points = np.array([polygon["all_points_x"], polygon["all_points_y"]]).T

    # Approximate the polygon with the given tolerance to smooth its borders.
    smoothed_points = skimage.measure.approximate_polygon(points, config.POLYGON_SMOOTHING_TOLERANCE)

    # Convert the smoothed points back to integer coordinates.
    smoothed_polygon = {
        "all_points_x": smoothed_points[:, 0].astype(int).tolist(),
        "all_points_y": smoothed_points[:, 1].astype(int).tolist()
    }

    return smoothed_polygon

def mask_background(image: np.ndarray, polygon: dict, config: Config) -> np.ndarray:
    """
    Masks the background of an image outside a specified polygon.

    This function takes an input image and a polygon defined by its x and y coordinates. It creates a mask
    with the polygon area filled (set to 1) and replaces all pixels outside the polygon with a default gray value.
    For a color image (3D), the replacement color is (128, 128, 128); for a grayscale image (2D), the replacement is 128.

    Parameters:
        image (np.ndarray): The input image, which can be either a color image (3D array) or a grayscale image (2D array).
        polygon (dict): A dictionary containing two keys, "all_points_x" and "all_points_y", each mapping to a list of 
                        integers that define the polygon vertices along the x-axis and y-axis respectively.

    Returns:
        np.ndarray: A new image array where all pixels outside the defined polygon are replaced by the default gray value.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if config.MASK_POLYGON_SMOOTHING:
        polygon = smooth_polygon(polygon, config)
    
    all_points_x = np.array(polygon["all_points_x"], dtype=np.int32)
    all_points_y = np.array(polygon["all_points_y"], dtype=np.int32)
    rr, cc = skimage.draw.polygon(all_points_y, all_points_x, shape=mask.shape)
    mask[rr, cc] = 1

    # Make a copy of the original image to apply the mask
    masked_image = image.copy()

    # Define the gray color using a default threshold value of 128
    mask_value = config.MASK_VALUE
    mask_color = (mask_value, mask_value, mask_value) if image.ndim == 3 else mask_value

    # Replace pixels outside the polygon (mask value != 1) with gray
    if image.ndim == 3:
        masked_image[mask != 1] = mask_color
    else:
        masked_image[mask != 1] = mask_color

    return masked_image

def convert_annotations(input_annotations: dict, target_class: str=Config.TARGET_CLASS) -> dict:
    """
    Converts a dictionary retrieved from a json of complete toothwise semantic annotations to 
    a new dictionary where keys are the tooth's label_id. and contain information about each tooth's image,
    number, polygon label, and a target class label. This forms the master json file for the toothwise semantic 
    segmentation, and has more information like the tooth polygon that can be used to mask the background and provide
    an attention area for the model.
    
    Each output entry has the form:
    {
      'tooth_img_name': <tooth_image_filename>,
      'tooth_number': <tooth_number>,
      'target_class': class name/ no_ ,
      'tooth_poly': {
          'all_points_x': [...],
          'all_points_y': [...]
      },
      ...
    }
    
    Args:
        input_annotations: A dictionary containing dataset annotations.
        
    Returns:
        A dictionary keyed by each tooth's label_id with the new structure.
    """
    output = {}
    for entry in input_annotations.values():
        teeth_data = entry.get("teeth_data", {})
        for tooth in teeth_data.values():
            label_id = tooth.get("label_id")
            if not label_id:
                continue
            
            # Extract basic info
            tooth_img_name = tooth.get("tooth_image_filename", "")
            tooth_number = tooth.get("tooth_number", None)
            
            # Check diagnostic_labels for plaque presence.
            class_value = 'no_'
            diag_labels = tooth.get("diagnostic_labels", {})
            for diag in diag_labels.values():
                # assuming region_class equal to "plaque" marks a positive plaque label
                if diag.get("region_class", "").lower() == target_class.lower():
                    class_value = target_class
                    break
            
            # Extract tooth_label (only x and y coordinates)
            tooth_label = tooth.get("tooth_label", {})
            filtered_tooth_label = {
                "all_points_x": tooth_label.get("all_points_x", []),
                "all_points_y": tooth_label.get("all_points_y", [])
            }
            
            output[label_id] = {
                "tooth_img_name": tooth_img_name,
                "tooth_number": tooth_number,
                "target_class": class_value,
                "tooth_poly": filtered_tooth_label
            }
    return output

def pad_and_resize(image: np.ndarray, target_dim: int=Config.TARGET_DIM,
                   mask_value: int=Config.MASK_VALUE) -> np.ndarray:
    """
    Pads an input image to be square using the largest image dimension,
    then resizes it to a square image with dimensions (target_dim x target_dim).

    Parameters:
        image (np.ndarray): Input image array (grayscale or color).
        target_dim (int): The desired dimension in pixels for the output square image. It is from the Config class.
        mask_value (int): The value to use for padding. It is from the Config class. It defines the masked area colour.

    Returns:
        np.ndarray: The padded and resized image.
    """
    h, w = image.shape[:2]
    max_side = max(h, w)
    
    # Calculate required padding amounts
    pad_height = max_side - h
    pad_width = max_side - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Determine correct constant value based on image dtype and range.
    if image.dtype == np.uint8:
        pad_const = mask_value  # value in 0-255
    elif np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        pad_const = mask_value / 255.0  # convert 128 to equivalent in [0,1]
    else:
        pad_const = mask_value  # fall back; adjust as needed
    
    # Pad image differently based on its dimensionality (grayscale vs. color)
    if image.ndim == 3:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values= pad_const)
    elif image.ndim == 2:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values= pad_const)
    else:
        raise ValueError("Unsupported image dimensions.")
    
    # Resize the padded image to target_dim x target_dim.
    # resized = tf.image.resize(image, (target_dim, target_dim), antialias= Config.ANTIALIZING_IN_RESIZING)
    # resized = resized / 255.0  # scale to [0, 1]
    # The resize function returns a float image in [0,1]. If the input is of type uint8, rescale to 0-255.
    resized = transform.resize(padded, (target_dim, target_dim), anti_aliasing= Config.ANTIALIZING_IN_RESIZING)
    if image.dtype == np.uint8:
        resized = (resized * 255).astype(np.uint8)
    
    return resized

def rescale_image(image: np.ndarray, config) -> np.ndarray:
    """
    Rescales an image's pixel values to a target range defined in config.rescale_pixels.
    Also, if the image's dtype differs from config.IMAGE_PVALUE_TYPE, it is converted.

    The function first determines if the input image is in the [0,1] or [0,255] range by 
    checking the maximum pixel value. It then applies a linear transformation so that the 
    image values map to the target range (e.g. [0,1], [-1,1], or [0,255]).

    Parameters:
        image (np.ndarray): The input image.
        config: A configuration object containing:
            - rescale_pixels: A tuple/list with two numbers (target_min, target_max).
            - IMAGE_PVALUE_TYPE: The desired numpy dtype for the image (e.g., np.uint8 or np.float32).

    Returns:
        np.ndarray: The rescaled image with pixel values in the target range and of type config.IMAGE_PVALUE_TYPE.
    """
    # Convert image type if necessary.
    if image.dtype != config.IMAGE_PVALUE_TYPE:
        image = image.astype(config.IMAGE_PVALUE_TYPE)

    # Determine input range.
    # If the maximum value is <= 1, assume the image range is [0, 1]; otherwise [0, 255].
    if image.max() <= 1.0:
        input_min, input_max = 0.0, 1.0
    else:
        input_min, input_max = 0.0, 255.0

    target_min, target_max = config.RESCALE_PIXELS

    # Clip the image to the detected input range.
    image = np.clip(image, input_min, input_max)

    # Avoid division by zero for constant images.
    if input_max == input_min:
        return np.full(image.shape, target_min, dtype=config.IMAGE_PVALUE_TYPE)

    # Apply linear transformation to map input range to target range.
    scaled_image = (image - input_min) / (input_max - input_min) * (target_max - target_min) + target_min

    # If target type is integer, round the values before casting.
    if config.IMAGE_PVALUE_TYPE in [np.uint8, np.int32, np.int16]:
        scaled_image = np.round(scaled_image).astype(config.IMAGE_PVALUE_TYPE)
    else:
        scaled_image = scaled_image.astype(config.IMAGE_PVALUE_TYPE)

    return scaled_image

def preprocess_record(
    record: Dict, # dict of {'image_path': image_path,'poly': tooth['bbox'], 'label': tooth['label'],'augmentation_label': False}
    config: Config,
) -> Tuple[np.ndarray, int]:
    """
        Preprocess a single record in the tooth classification data pipeline.
        This function loads an image specified by the input record, applies various
        preprocessing steps such as white-balancing, background masking, padding/resizing,
        and optional pixel rescaling based on the configuration provided. The input record
        is expected to be a tuple containing the image filename, its corresponding tooth
        polygon/mask, and an associated label.
        Parameters:
            record (Tuple[str, Dict, int]): A tuple containing:
                - image_filename (str): The filename of the image to be processed.
                - tooth_poly (Dict): A dictionary representing the tooth polygon/mask.
                - label (int): The label associated with the image.
            config (Config): A configuration object that includes preprocessing settings, 
                            notably for pixel rescaling (e.g., in the attribute RESCALE_PIXELS).
        Returns:
            Tuple[np.ndarray, int]: A tuple containing the preprocessed image as a NumPy array
            and its corresponding label.
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be read.
    """
    
    # IMAGE PREPROCESSING
    img_name = record[0]
    img_path = os.path.join(config.DATA_DIR, config.IMAGE_DIR, img_name)
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    image = io.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    
    if config.NORMALIZE_IMAGES:
        image = dental_gray_world_white_balance(image)
    # Mask
    if config.MASK_BG:
        image = mask_background(image, record[1])
    
    # Pad and resize
    image = pad_and_resize(image, target_dim=config.TARGET_DIM, mask_value= config.MASK_VALUE)
    
    #rescale:
    if config.RESCALE_PIXELS[0] is not None:
        image = rescale_image(image, config)

    return image, record[2]

def preprocess_dataset(
    json_annotations_path: str,#: pd.DataFrame,
    output_dir: str,
    image_dir: str = Config.IMAGE_DIR,
    target_dim: int = Config.TARGET_DIM,
    mask_bg: bool = Config.MASK_BG,
    mask_value: int = Config.MASK_VALUE,
    verbose: bool = False,
    target_class_name: str = Config.TARGET_CLASS,
    remove_dark_images: bool = Config.REMOVE_DARK_IMAGES,
    normalize_images: bool = Config.NORMALIZE_IMAGES
) -> None:
    """
    Goes through each annotation, loads image, pads and resizes,
    optionally masks the background, and saves the final image to output_dir 
    with the appropriate label in the filename or subfolder.

    Example: output_dir/<label>/image_filename.jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    
    #loading the raw json file
    with open(json_annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f) 
    
    #filtering out the dark images
    if remove_dark_images:
        if verbose: print("Filtering out dark images...")
        annotations = remove_dark_images_from_json(annotations, image_dir)
        print(f"Number of filtered annotations: {len(annotations)}")
        if verbose: print("Saving filtered annotations...")
    
    #converting the raw json file to the format required for the target class
    if verbose: print("Converting annotations...")
    annotations = convert_annotations(annotations)
    print(f"Number of annotations: {len(annotations)}")

    annotations = convert_int32_to_int(annotations)
    with open(os.path.join(output_dir,f"filtered_{target_class_name}_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    
    #initiating a progress bar
    progress_bar = tqdm(total=len(annotations), desc="Processing images", unit="image")
    
    # IMAGE PREPROCESSING
    # if verbose: print("Processing images ...")
    for idx, record in annotations.items():
        
        img_name = record["tooth_img_name"]
        img_path = os.path.join(image_dir,img_name)
        
        if not os.path.exists(img_path):
            continue
        
        image = io.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}. jumping to next image...")
            continue
        if normalize_images:
            image = dental_gray_world_white_balance(image)
        # Mask
        if mask_bg:
            image = mask_background(image, record["tooth_poly"])
        
        # Pad and resize
        image = pad_and_resize(image, target_dim=target_dim, mask_value= mask_value)
        
        # Construct output filename
        out_path = os.path.join(output_dir, img_name)
        
        io.imsave(out_path, image)
        # print(f"Saved: {out_path}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    if verbose: print("Done pre-processing the dataset.")
    
def split_dataset_json(master_json_path: str,
                       output_dir: str,
                       train_ratio: float = Config.TRAIN_RATIO,
                       val_ratio: float = Config.VAL_RATIO,
                       test_ratio: float = Config.TEST_RATIO,
                       random_seed: int = Config.RANDOM_SEED) -> tuple:
    """
    Loads the master dataset JSON and splits it into train, validation, and test sets stratified by target_class.
    Writes three JSON files to output_dir with the same structure as the master JSON.
    
    Args:
        master_json_path (str): Path to the master JSON dataset.
        output_dir (str): Directory where the split JSON files will be saved.
        train_ratio (float): Fraction of records for training.
        val_ratio (float): Fraction of records for validation.
        test_ratio (float): Fraction of records for testing.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_json, val_json, test_json) dictionaries.
    """

    # Load the master JSON data (assumed to be a dictionary keyed by record IDs)
    with open(master_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a list of record IDs and corresponding target classes.
    records = list(data.items())  # Each item is a tuple (record_id, record)
    record_ids = [rec_id for rec_id, rec in records]
    target_classes = [rec["target_class"] for rec_id, rec in records]
    
    # First, split off the training set.
    train_ids, temp_ids, _, temp_targets = train_test_split(
        record_ids, target_classes,
        test_size=(1-train_ratio),
        random_state=random_seed,
        stratify=target_classes
    )
    
    # Among the remaining records, split them into validation and test.
    # Compute the fraction of val from the remaining records.
    val_frac = val_ratio / (val_ratio + test_ratio)
    
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_targets,
        test_size=(1 - val_frac),
        random_state=random_seed,
        stratify=temp_targets
    )
    
    # Build dictionaries for each split preserving the same structure.
    train_json = {rec_id: data[rec_id] for rec_id in train_ids}
    val_json   = {rec_id: data[rec_id] for rec_id in val_ids}
    test_json  = {rec_id: data[rec_id] for rec_id in test_ids}
    
    # Ensure output directory exists and save the splits to JSON files.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{Config.TARGET_CLASS}_train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(output_dir, f"{Config.TARGET_CLASS}_val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_json, f, indent=2)
    with open(os.path.join(output_dir, f"{Config.TARGET_CLASS}_test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_json, f, indent=2)
    
    return train_json, val_json, test_json



def build_tf_dataset(
    records: Dict[str, dict],
    batch_size: int = Config.BATCH_SIZE,
    shuffle: bool = Config.SHUFFLE_DATASET,
    augment: bool = Config.AUGMENT_DATA,
    image_size: tuple[int, int] = (Config.TARGET_DIM, Config.TARGET_DIM),
    target_class_name: str = Config.TARGET_CLASS,
    data_parent_dir: str = Config.DATA_DIR,
    image_dir: str = Config.IMAGE_DIR,
    normalize_images: bool = Config.NORMALIZE_IMAGES
) -> tf.data.Dataset:
    """
    Builds a TensorFlow Dataset from a list of (image_path, label) tuples.
    Optionally apply data augmentation.

    Example of usage:
        train_ds = build_tf_dataset(train_samples)
    """
    
    def _load_and_preprocess(image_path, label):
        # Read image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.image.resize(image, image_size)  # or any dimension
        # image = image / 255.0  # scale to [0, 1]
        
        # Mask the background using the helper function if requested.
        # Wrap the NumPy-based mask_background() into a TF op using tf.numpy_function.
        # if mask_bg:
        #     # Call mask_background() with image and tooth_poly.
        #     # Note: tooth_poly is assumed to be a dict containing keys "all_points_x" and "all_points_y".
        #     image = tf.numpy_function(
        #         func=lambda img, poly: mask_background(img, poly),
        #         inp=[image, tooth_poly],
        #         Tout=image.dtype
        #     )
        #     # Propagate the image shape information.
        #     image.set_shape((Config.TARGET_DIM, Config.TARGET_DIM, 3))
        if normalize_images:
            image = tf.numpy_function(func= lambda img: dental_gray_world_white_balance(img), inp=[image], Tout=image.dtype)
        
        # Optional: augment
        if augment:
            # simple horizontal flip
            image = tf.image.random_flip_left_right(image)
            # maybe random brightness
            image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Convert label (string) to an integer class if needed
        # For instance, if you have "pla" -> 1, "no_plaque" -> 0
        label_int = tf.cond(
            tf.math.equal(label, tf.constant(target_class_name)),
            lambda: tf.constant(1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )
        
        return image, label_int

    # Convert the samples to a TF Dataset
    paths = []
    labels = []
    for key, record in records.items():
        paths.append(os.path.join(os.path.join(data_parent_dir, image_dir), record["tooth_img_name"]))
        labels.append(record["target_class"])
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(records), reshuffle_each_iteration=True)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

if __name__ == "__main__":
    """
    If this file is run as a script (e.g., python data_pipeline.py),
    you might put a small demo or test code here.
    """
    pass
