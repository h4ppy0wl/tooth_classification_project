# data_pipeline.py

import os
import sys
import random
import json
from typing import List, Tuple, Dict
import numpy as np
import datetime
import cv2
import skimage
from skimage import io, transform, draw, color
from skimage.io import imsave
from skimage.filters import gaussian
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(current_dir)
from src.config import Config

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

@tf.function
def tf_dental_gray_world_white_balance(image_rgb):
    """
    A preprocessing function to apply modified gray-world white balance for dental images.
    Preserves red (gums/tongue) while balancing white (teeth).

    Args:
        image_rgb: A TensorFlow tensor representing the RGB image. Expected dtype is either tf.uint8 or tf.float32.
                   If tf.uint8, the image is normalized to [0,1].

    Returns:
        A TensorFlow tensor with white-balanced image values in the range [0,1].
    """
    # Convert to float for processing if needed.
    # (tf.uint8 images will be cast and normalized; otherwise, assume already in float format.)
    if image_rgb.dtype == tf.uint8:
        img_float = tf.cast(image_rgb, tf.float32) / 255.0
    else:
        img_float = tf.identity(image_rgb)
    
    # Convert to HSV. TensorFlow's rgb_to_hsv expects the input image in [0,1].
    img_hsv = tf.image.rgb_to_hsv(img_float)
    
    # Create a "blue mask" (glove mask) in the HSV space.
    # blue_mask = ((H > 0.43) & (H < 0.70)) & ((S > 0.30) & (S < 0.55))
    mask_hue_blue = tf.logical_and(tf.greater(img_hsv[..., 0], 0.43),
                                   tf.less(img_hsv[..., 0], 0.70))
    mask_sat_blue = tf.logical_and(tf.greater(img_hsv[..., 1], 0.30),
                                   tf.less(img_hsv[..., 1], 0.55))
    blue_mask = tf.logical_and(mask_hue_blue, mask_sat_blue)
    
    # Create a "black mask" for dark regions in the image.
    # black_mask = ((H > 0.95) or (H < 0.05)) & (S > 0.55)
    mask_hue_black = tf.logical_or(tf.greater(img_hsv[..., 0], 0.95),
                                   tf.less(img_hsv[..., 0], 0.05))
    mask_sat_black = tf.greater(img_hsv[..., 1], 0.55)
    black_mask = tf.logical_and(mask_hue_black, mask_sat_black)
    # Explicitly set the shape for the mask.
    black_mask.set_shape([None, None])
    
    # Compute the mean of the V channel over non-black pixels.
    non_black_v = tf.boolean_mask(img_hsv[..., 2], tf.logical_not(black_mask))
    avg_v = tf.reduce_mean(non_black_v)
    
    # For RGB channel averages, exclude pixels in either blue_mask or black_mask.
    combined_mask = tf.logical_not(tf.logical_or(blue_mask, black_mask))
    combined_mask.set_shape([None, None])
    avg_r = tf.reduce_mean(tf.boolean_mask(img_float[..., 0], combined_mask))
    avg_g = tf.reduce_mean(tf.boolean_mask(img_float[..., 1], combined_mask))
    avg_b = tf.reduce_mean(tf.boolean_mask(img_float[..., 2], combined_mask))
    
    # Compute global gray mean.
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    factor_scaler = 2.0
    factor = 0.8 + factor_scaler * tf.square(avg_v - 1.0)
    
    # Apply channel-wise correction.
    red_corrected = img_float[..., 0] * ((factor + 0.2) * (avg_gray / avg_r))
    green_corrected = img_float[..., 1] * (factor * (avg_gray / avg_g))
    blue_corrected = img_float[..., 2] * (factor * (avg_gray / avg_b))
    
    # Reconstruct the image.
    img_corrected = tf.stack([red_corrected, green_corrected, blue_corrected], axis=-1)
    
    # Clip values to the range [0,1].
    img_corrected = tf.clip_by_value(img_corrected, 0.0, 1.0)
    
    return img_corrected

def is_darker_than_threshold(image_path: str, threshold: float):
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

def remove_dark_images_from_json(json_data: dict, image_base_path: str, threshold) -> dict:
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
    dark_images = []
    progress_bar = tqdm(total=len(json_data), desc="removing dark images", unit="oral cavity image")
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
            if is_darker_than_threshold(full_image_path, threshold):  # Skip dark images
                dark_images.append(tooth_image_filename)
                # print(f"{tooth_image_filename} was dark, removed")
                continue
            filtered_teeth[tooth_key] = tooth_entry
        # Only add entry if at least one tooth passed our brightness check
        if len(filtered_teeth) > 0:
            entry["teeth_data"] = filtered_teeth
            filtered_data[key] = entry
        progress_bar.update(1)
        
    progress_bar.close()
    return filtered_data, dark_images

def parse_dataset_json(json_path: str, config: Config, is_train_ds = True) -> list:

    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Expecting a list of dicts

    # Build a list of base records (augmentation_label=False by default)
    records = []
    for item in data.values():  # each item is one full image
        r = [
            item['tooth_img_name'],#'image_name': 
            item['tooth_poly']['all_points_x'],#'tooth_polygon_x': 
            item['tooth_poly']['all_points_y'],#'tooth_polygon_y': 
            item['target_class'], # 'label':  "no_" or config.TARGET_CLASS (e.g., "Pla")
            False #'augmentation_label': 
        ]
        records.append(r)

    if (config.OVERSAMPLE_FACTOR == 1) or not (is_train_ds) :
        random.shuffle(records)
        return records
    # Separate target_class  from the other
    target_class_records = [r for r in records if r[3] == config.TARGET_CLASS]
    # Optionally, you can also do: without_plaque_records = [r for r in records if r['label'] != config.TARGET_CLASS]

    # 4) Create extra copies of the minority (with plaque) records 
    #    and set augmentation_label=True for them
    oversampled_records = []
    # E.g., if oversample_factor=3, each with-plaque record is repeated 2 additional times
    for r in target_class_records:
        for _ in range(config.OVERSAMPLE_FACTOR - 1):
            new_record = r.copy()
            new_record[-1] = True
            oversampled_records.append(new_record)

    # 5) Merge original + extra augmented records
    all_records = records + oversampled_records

    # 6) Shuffle so the augmented records are interspersed randomly
    random.shuffle(all_records)

    return all_records #list of dictionaries

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

def resize_and_mask_background(image: np.ndarray, polygon: list, config: Config) -> np.ndarray:
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
    mask = np.zeros((height, width), dtype=bool)
    
    if config.MASK_POLYGON_SMOOTHING:
        polygon = smooth_polygon(polygon, config)
    
    all_points_x = np.array(polygon[0], dtype=np.int32)
    all_points_y = np.array(polygon[1], dtype=np.int32)
    rr, cc = skimage.draw.polygon(all_points_y, all_points_x, shape=mask.shape)
    mask[rr, cc] = True


    # Determine the scaling factor.
    dmax = max(height, width)
    scale = config.TARGET_DIM / dmax
    if dmax == height:
        new_h = config.TARGET_DIM
        new_w = int(round(width * scale))
    else:
        new_h = int(round(height * scale))
        new_w = config.TARGET_DIM
    if scale > 1:
        resized_img = transform.resize(image, (new_h, new_w), preserve_range= True, anti_aliasing= False, order=1)
    if scale <= 1:
        resized_img = transform.resize(image, (new_h, new_w), preserve_range= True, anti_aliasing= config.ANTIALIZING_IN_RESIZING, order=1)
    
    resized_mask = transform.resize(mask, (new_h, new_w), preserve_range= True, anti_aliasing= False, order=0)
    resized_mask = (resized_mask > 0.5)
    # Define the gray color using a default threshold value of 128
    mask_value = config.MASK_VALUE
    if not config.MASK_BG:
        return resized_img
    if mask_value in range(5,50):
        float_img = resized_img.copy()
        if resized_img.max() > 2:
            # Convert image to float [0,1] for skimage
            float_img = resized_img.astype(np.float32) / 255.0

        # Apply keras.filters.gaussian blur to the entire image
        # 'multichannel=True' ensures the filter is applied per channel
        blurred = gaussian(float_img, sigma=mask_value, multichannel=True)

        # Combine: inside polygon = original; outside polygon = blurred
        # (mask is [H,W], but broadcasting works for color channels)
        out = blurred.copy()
        out[resized_mask] = float_img[resized_mask]

        # Convert back to [0..255]
        masked_image = out.astype(config.IMAGE_PVALUE_TYPE)
    else:
        # Make a copy of the original image to apply the mask
        masked_image = resized_img.copy()
        if image.dtype == np.uint8:
            mask_color = (mask_value, mask_value, mask_value) if image.ndim == 3 else mask_value
        if image.dtype == np.float32 and image.max() < 2:
            mask_color = (mask_value/255.0, mask_value/255.0, mask_value/255.0) if image.ndim == 3 else mask_value/255.0

        # Replace pixels outside the polygon (mask value != 1) with gray
        masked_image[~resized_mask] = mask_color
        # if image.ndim == 3:
        #     masked_image[mask != 1] = mask_color
        # else:
        #     masked_image[mask != 1] = mask_color


    return masked_image

# def tf_gaussian_kernel(size, sigma):
#     """
#     Creates a 1D Gaussian kernel.
#     """
#     x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
#     kernel = tf.exp(-0.5 * tf.square(x / sigma))
#     kernel /= tf.reduce_sum(kernel)
#     return kernel

# # Global variable to hold the cached kernel.
# CACHED_GAUSSIAN_KERNEL = None

# def init_cached_gaussian_kernel(config):
#     """Initializes the global cached Gaussian kernel if needed."""
#     global CACHED_GAUSSIAN_KERNEL
#     sigma = tf.cast(config.MASK_VALUE, tf.float32)
#     # Kernel size: 2*ceil(3*sigma)+1
#     kernel_size = tf.cast(2 * tf.math.ceil(3 * sigma) + 1, tf.int32)
#     kernel_1d = tf_gaussian_kernel(kernel_size, sigma)  # This function must return a 1D kernel.
#     kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
#     # Expand dimensions so that shape is (kernel_size, kernel_size, 1, 1)
#     CACHED_GAUSSIAN_KERNEL = kernel_2d[:, :, tf.newaxis, tf.newaxis]



# @tf.function
# def tf_mask_background(image, polygon, config):
#     """
#     Masks the background of an image outside a specified polygon.
    
#     For color images, pixels outside the polygon are replaced by a gray value
#     (mask_value, mask_value, mask_value). For grayscale images, the replacement is mask_value.
#     When config.MASK_VALUE is between 5 and 50, a Gaussian blur is applied to the outside region.
    
#     Args:
#         image: A TensorFlow tensor representing the image (H x W x C or H x W).
#         polygon: A tuple or list of two elements, where polygon[0] and polygon[1] are lists of
#                  x and y coordinates of the polygon vertices.
#         config: A configuration object with attributes:
#             - MASK_POLYGON_SMOOTHING (bool)
#             - MASK_VALUE (int)
#             - IMAGE_PVALUE_TYPE (tf.DType) e.g. tf.uint8
        
#     Returns:
#         A tensor with the same shape as image with background masked.
#     """
#     # Get image height and width.
#     shape = tf.shape(image)
#     height = shape[0]
#     width = shape[1]

#     # Optionally smooth the polygon.
#     if config.MASK_POLYGON_SMOOTHING:
#         polygon = smooth_polygon(polygon, config)  # Assumed to be TF-compatible.

#     # Convert polygon coordinate lists to tensors of type float32.
#     poly_x = tf.cast(polygon[0], tf.float32)  # shape (n_points,)
#     poly_y = tf.cast(polygon[1], tf.float32)  # shape (n_points,)

#     # Create a meshgrid of pixel coordinates.
#     x_range = tf.cast(tf.range(width), tf.float32)
#     y_range = tf.cast(tf.range(height), tf.float32)
#     grid_x, grid_y = tf.meshgrid(x_range, y_range)  # both shape (height, width)

#     # Expand dims so that grid coordinates can be compared to each polygon vertex.
#     grid_x_exp = tf.expand_dims(grid_x, axis=-1)  # (H, W, 1)
#     grid_y_exp = tf.expand_dims(grid_y, axis=-1)  # (H, W, 1)

#     # Reshape polygon coordinates for broadcasting.
#     poly_x_exp = tf.reshape(poly_x, [1, 1, -1])  # (1, 1, n_points)
#     poly_y_exp = tf.reshape(poly_y, [1, 1, -1])  # (1, 1, n_points)

#     # Also get the "next" vertex for each edge (using tf.roll).
#     poly_x_next = tf.roll(poly_x, shift=-1, axis=0)
#     poly_y_next = tf.roll(poly_y, shift=-1, axis=0)
#     poly_x_next_exp = tf.reshape(tf.cast(poly_x_next, tf.float32), [1, 1, -1])
#     poly_y_next_exp = tf.reshape(tf.cast(poly_y_next, tf.float32), [1, 1, -1])

#     # Compute conditions for the ray-casting algorithm.
#     # Condition 1: The y-coordinate of the point is between the y's of the edge endpoints.
#     cond1 = tf.math.not_equal(poly_y_exp > grid_y_exp, poly_y_next_exp > grid_y_exp)
#     # Compute the x-coordinate at which the horizontal line at grid_y intersects the edge.
#     epsilon = 1e-6
#     slope = (poly_x_next_exp - poly_x_exp) / (poly_y_next_exp - poly_y_exp + epsilon)
#     intersect_x = slope * (grid_y_exp - poly_y_exp) + poly_x_exp
#     cond2 = grid_x_exp < intersect_x

#     # An edge is crossed if both conditions hold.
#     crossings = tf.logical_and(cond1, cond2)
#     # Count the number of crossings per pixel.
#     crossing_count = tf.reduce_sum(tf.cast(crossings, tf.int32), axis=-1)
#     # Inside polygon if count is odd.
#     mask = tf.math.mod(crossing_count, 2) == 1  # shape (height, width)

#     # Get the mask value from the configuration.
#     mask_value = config.MASK_VALUE  # assumed to be a python integer

#     # Case 1: When mask_value is in the range 5 to 50, apply Gaussian blur to the outside region.
#     if mask_value in range(5, 50):
#         # # Ensure the image is in float [0,1].
#         # if image.dtype == tf.uint8:
#         #     image_float = tf.cast(image, tf.float32) / 255.0
#         # else:
#         #     image_float = tf.identity(image)

#         # sigma = tf.cast(mask_value, tf.float32)
#         # # Determine kernel size: 2*ceil(3*sigma)+1
#         # kernel_size = tf.cast(2 * tf.math.ceil(3 * sigma) + 1, tf.int32)
#         # kernel_1d = tf_gaussian_kernel(kernel_size, sigma)  # (kernel_size,)
#         # # Create 2D separable kernel.
#         # kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)  # (kernel_size, kernel_size)
#         # kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]  # (k, k, 1, 1)

#         # # Apply convolution based on image rank.
#         # if tf.rank(image_float) == 3 and tf.shape(image_float)[-1] == 3:
#         #     # For color images, apply the same kernel to each channel using depthwise convolution.
#         #     channels = 3
#         #     kernel_2d = tf.tile(kernel_2d, [1, 1, channels, 1])
#         #     image_exp = tf.expand_dims(image_float, axis=0)  # add batch dim
#         #     blurred = tf.nn.depthwise_conv2d(image_exp, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')
#         #     blurred = tf.squeeze(blurred, axis=0)
#         # else:
#         #     # For grayscale images.
#         #     image_exp = tf.expand_dims(tf.expand_dims(image_float, axis=0), axis=-1)  # shape (1,H,W,1)
#         #     blurred = tf.nn.conv2d(image_exp, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')
#         #     blurred = tf.squeeze(blurred, axis=[0, -1])
        
#         # Ensure the image is in float [0,1]
#         if image.dtype == tf.uint8:
#             image_float = tf.cast(image, tf.float32) / 255.0
#         else:
#             image_float = tf.identity(image)

#         sigma = tf.cast(mask_value, tf.float32)
#         # # Determine kernel size: 2*ceil(3*sigma)+1
#         # kernel_size = tf.cast(2 * tf.math.ceil(3 * sigma) + 1, tf.int32)
#         # kernel_1d = tf_gaussian_kernel(kernel_size, sigma)  # shape (kernel_size,)
#         # Create a 2D separable kernel.
#         kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)  # shape (kernel_size, kernel_size)
#         kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]  # shape (k, k, 1, 1)
#         pad_amt = kernel_size // 2

#         # Use the cached kernel instead of computing it here.
#         global CACHED_GAUSSIAN_KERNEL
#         kernel_2d = CACHED_GAUSSIAN_KERNEL  # shape: (kernel_size, kernel_size, 1, 1)
#         pad_amt = tf.cast(tf.shape(kernel_2d)[0] // 2, tf.int32)


        
#         # # Instead of computing a large Gaussian kernel, we approximate the blur.
#         # # Expand dims to add a batch dimension.
#         # image_exp = tf.expand_dims(image_float, axis=0)
#         # blurred = approximate_gaussian_blur(image_exp, num_passes=3)
#         # blurred = tf.squeeze(blurred, axis=0)

#         # For color images:
#         if tf.rank(image_float) == 3 and tf.shape(image_float)[-1] == 3:
#             channels = 3
#             # Expand dims to add a batch dimension.
#             image_exp = tf.expand_dims(image_float, axis=0)
#             # Pad using reflection to avoid zero-edge artifacts.
#             image_padded = tf.pad(
#                 image_exp,
#                 paddings=[[0, 0], [pad_amt, pad_amt], [pad_amt, pad_amt], [0, 0]],
#                 mode="REFLECT"
#             )
#             # Apply depthwise convolution with VALID padding.
#             blurred = tf.nn.depthwise_conv2d(image_padded, 
#                                             tf.tile(kernel_2d, [1, 1, channels, 1]), 
#                                             strides=[1, 1, 1, 1], 
#                                             padding='VALID')
#             blurred = tf.squeeze(blurred, axis=0)
#         else:
#             # For grayscale images:
#             image_exp = tf.expand_dims(tf.expand_dims(image_float, axis=0), axis=-1)
#             image_padded = tf.pad(
#                 image_exp,
#                 paddings=[[0, 0], [pad_amt, pad_amt], [pad_amt, pad_amt], [0, 0]],
#                 mode="REFLECT"
#             )
#             blurred = tf.nn.conv2d(image_padded, kernel_2d, strides=[1, 1, 1, 1], padding='VALID')
#             blurred = tf.squeeze(blurred, axis=[0, -1])
        
        
        
        
        
#         # Combine: inside polygon, keep original; outside, use blurred.
#         # Expand mask for broadcasting.
#         if tf.rank(image_float) == 3:
#             mask_exp = tf.cast(tf.expand_dims(mask, axis=-1), image_float.dtype)
#         else:
#             mask_exp = tf.cast(mask, image_float.dtype)
#         combined = image_float * mask_exp + blurred * (1 - mask_exp)
        
#         # Convert back to desired type.
#         if image.dtype == tf.uint8:
#             masked_image = tf.cast(tf.clip_by_value(combined * 255.0, 0, 255), config.IMAGE_PVALUE_TYPE)
#         else:
#             masked_image = tf.cast(tf.clip_by_value(combined, 0.0, 1.0), config.IMAGE_PVALUE_TYPE)
#     else:
#         # Case 2: No blurring; simply replace pixels outside the polygon with a gray value.
#         if tf.rank(image) == 3 and tf.shape(image)[-1] == 3:
#             # Use a Python if‑statement to check dtype.
#             if image.dtype == tf.uint8:
#                 mask_color = tf.constant([mask_value,
#                                         mask_value,
#                                         mask_value], dtype=image.dtype)
#             else:
#                 mask_color = tf.constant([mask_value / 255.0,
#                                         mask_value / 255.0,
#                                         mask_value / 255.0], dtype=image.dtype)
#             mask_color_img = tf.ones_like(image) * mask_color
#         else:
#             mask_color_img = tf.ones_like(image) * (
#                 mask_value if image.dtype == tf.uint8 else mask_value / 255.0
#             )
#         # Use tf.where to select pixels: if inside polygon, keep original.
#         if tf.rank(image) == 3:
#             mask_exp = tf.expand_dims(mask, axis=-1)
#         else:
#             mask_exp = mask
#         masked_image = tf.where(mask_exp, image, mask_color_img)

#     return masked_image

@tf.function
def approximate_gaussian_blur(image, num_passes=3):
    # image is assumed to be of shape [1, H, W, C] (i.e. with a batch dimension)
    for _ in range(num_passes):
        image = tf.nn.avg_pool2d(image, ksize=20, strides=1, padding='SAME')
    return image

@tf.function
def tf_mask_background(image, polygon, config):
    """
    Masks the background of an image outside a specified polygon.
    
    For color images, pixels outside the polygon are replaced by a gray value
    (mask_value, mask_value, mask_value). For grayscale images, the replacement is mask_value.
    When config.MASK_VALUE is between 5 and 50, a Gaussian blur is applied to the outside region.
    
    Args:
        image: A TensorFlow tensor representing the image (H x W x C or H x W).
        polygon: A tuple or list of two elements, where polygon[0] and polygon[1] are lists of
                 x and y coordinates of the polygon vertices.
        config: A configuration object with attributes:
            - MASK_POLYGON_SMOOTHING (bool)
            - MASK_VALUE (int)
            - IMAGE_PVALUE_TYPE (tf.DType) e.g. tf.uint8
        
    Returns:
        A tensor with the same shape as image with background masked.
    """
    # Get image height and width.
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    # Optionally smooth the polygon.
    if config.MASK_POLYGON_SMOOTHING:
        polygon = smooth_polygon(polygon, config)  # Assumed to be TF-compatible.

    # Convert polygon coordinate lists to tensors of type float32.
    poly_x = tf.cast(polygon[0], tf.float32)  # shape (n_points,)
    poly_y = tf.cast(polygon[1], tf.float32)  # shape (n_points,)

    # Create a meshgrid of pixel coordinates.
    x_range = tf.cast(tf.range(width), tf.float32)
    y_range = tf.cast(tf.range(height), tf.float32)
    grid_x, grid_y = tf.meshgrid(x_range, y_range)  # both shape (height, width)

    # Expand dims so that grid coordinates can be compared to each polygon vertex.
    grid_x_exp = tf.expand_dims(grid_x, axis=-1)  # (H, W, 1)
    grid_y_exp = tf.expand_dims(grid_y, axis=-1)  # (H, W, 1)

    # Reshape polygon coordinates for broadcasting.
    poly_x_exp = tf.reshape(poly_x, [1, 1, -1])  # (1, 1, n_points)
    poly_y_exp = tf.reshape(poly_y, [1, 1, -1])  # (1, 1, n_points)

    # Also get the "next" vertex for each edge (using tf.roll).
    poly_x_next = tf.roll(poly_x, shift=-1, axis=0)
    poly_y_next = tf.roll(poly_y, shift=-1, axis=0)
    poly_x_next_exp = tf.reshape(tf.cast(poly_x_next, tf.float32), [1, 1, -1])
    poly_y_next_exp = tf.reshape(tf.cast(poly_y_next, tf.float32), [1, 1, -1])

    # Compute conditions for the ray-casting algorithm.
    # Condition 1: The y-coordinate of the point is between the y's of the edge endpoints.
    cond1 = tf.math.not_equal(poly_y_exp > grid_y_exp, poly_y_next_exp > grid_y_exp)
    # Compute the x-coordinate at which the horizontal line at grid_y intersects the edge.
    epsilon = 1e-6
    slope = (poly_x_next_exp - poly_x_exp) / (poly_y_next_exp - poly_y_exp + epsilon)
    intersect_x = slope * (grid_y_exp - poly_y_exp) + poly_x_exp
    cond2 = grid_x_exp < intersect_x

    # An edge is crossed if both conditions hold.
    crossings = tf.logical_and(cond1, cond2)
    # Count the number of crossings per pixel.
    crossing_count = tf.reduce_sum(tf.cast(crossings, tf.int32), axis=-1)
    # Inside polygon if count is odd.
    mask = tf.math.mod(crossing_count, 2) == 1  # shape (height, width)

    # Get the mask value from config.
    mask_value = config.MASK_VALUE  # Python integer

    # Case 1: When mask_value is in the range 5 to 50, apply (approximated) blur to the outside.
    if mask_value in range(5, 50):
        # Convert image to float if necessary.
        if image.dtype == tf.uint8:
            image_float = tf.cast(image, tf.float32) / 255.0
        else:
            image_float = tf.identity(image)
        
        # Instead of computing a large Gaussian kernel, we approximate the blur.
        # Expand dims to add a batch dimension.
        image_exp = tf.expand_dims(image_float, axis=0)
        blurred = approximate_gaussian_blur(image_exp, num_passes= mask_value)
        blurred = tf.squeeze(blurred, axis=0)
        
        # Combine: inside polygon use original, outside use blurred.
        if tf.rank(image_float) == 3:
            mask_exp = tf.cast(tf.expand_dims(mask, axis=-1), image_float.dtype)
        else:
            mask_exp = tf.cast(mask, image_float.dtype)
        combined = image_float * mask_exp + blurred * (1 - mask_exp)
        
        if image.dtype == tf.uint8:
            masked_image = tf.cast(tf.clip_by_value(combined * 255.0, 0, 255), config.IMAGE_PVALUE_TYPE)
        else:
            masked_image = tf.cast(tf.clip_by_value(combined, 0.0, 1.0), config.IMAGE_PVALUE_TYPE)
    else:
        # Case 2: No blurring; simply replace pixels outside polygon with a constant gray value.
        if tf.rank(image) == 3 and tf.shape(image)[-1] == 3:
            if image.dtype == tf.uint8:
                mask_color = tf.constant([mask_value, mask_value, mask_value], dtype=image.dtype)
            else:
                mask_color = tf.constant([mask_value/255.0, mask_value/255.0, mask_value/255.0], dtype=image.dtype)
            mask_color_img = tf.ones_like(image) * mask_color
        else:
            mask_color_img = tf.ones_like(image) * (mask_value if image.dtype == tf.uint8 else mask_value/255.0)
        if tf.rank(image) == 3:
            mask_exp = tf.expand_dims(mask, axis=-1)
        else:
            mask_exp = mask
        masked_image = tf.where(mask_exp, image, mask_color_img)
    
    return masked_image


def convert_annotations(input_annotations: dict, target_class: str) -> dict:
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
        'rles' : [[]]
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
            rle = []
            diag_labels = tooth.get("diagnostic_labels", {})
            for diag in diag_labels.values():
                # assuming region_class equal to "plaque" marks a positive plaque label
                if diag.get("region_class", "").lower() == target_class.lower():
                    class_value = target_class
                    rle.append(diag['rle'])
                
            
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
                "tooth_poly": filtered_tooth_label,
                "rles": rle
            }
    return output

def pad_and_resize(image: np.ndarray, target_dim: int, mask_value: int) -> np.ndarray:
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
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')#, constant_values= pad_const)
    elif image.ndim == 2:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')#, constant_values= pad_const)
    else:
        raise ValueError("Unsupported image dimensions.")
    
    # Resize the padded image to target_dim x target_dim.
    # resized = tf.image.resize(image, (target_dim, target_dim), antialias= Config.ANTIALIZING_IN_RESIZING)
    # resized = resized / 255.0  # scale to [0, 1]
    # The resize function returns a float image in [0,1]. If the input is of type uint8, rescale to 0-255.
    resized = transform.resize(padded, (target_dim, target_dim), anti_aliasing= Config.ANTIALIZING_IN_RESIZING, order=2)
    if image.dtype == np.uint8:
        resized = (resized * 255).astype(np.uint8)
    
    return resized


def pad_image(image: np.ndarray, target_dim: int, config: Config) -> np.ndarray:
    """
    Pads an input image to be square using constant padding.
    
    The image is padded so that its height and width become equal to the maximum
    of the original dimensions. The padding is filled with the mask_value (or
    mask_value/255.0 for floating point images with values in [0,1]).
    
    Parameters:
        image (np.ndarray): Input image array (grayscale or color).
        mask_value (int): The value to use for padding (e.g. 128 for gray).
    
    Returns:
        np.ndarray: The padded image.
    """
    h, w = image.shape[:2]
    max_side = target_dim #max(h, w)
    
    # Calculate required padding amounts.
    pad_height = max_side - h
    pad_width  = max_side - w
    pad_top    = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left   = pad_width // 2
    pad_right  = pad_width - pad_left
    
    # Determine the constant value for padding based on image dtype.
    if image.dtype == np.uint8:
        pad_const = config.MASK_VALUE  # e.g. 128
    elif np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        pad_const = config.MASK_VALUE / 255.0  # e.g. 128 -> ~0.5
    else:
        pad_const = config.MASK_VALUE  # fallback

    # Pad image using constant padding.
    if image.ndim == 3:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='edge',
            # constant_values=((pad_const, pad_const), (pad_const, pad_const), (0, 0))
        )
    elif image.ndim == 2:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='edge',
            # constant_values=pad_const
        )
    else:
        raise ValueError("Unsupported image dimensions.")
    
    if not config.MASK_BG:
        # Create a blurred version of the padded image using skimage's gaussian filter.
        blurred = skimage.filters.gaussian(padded, sigma=10, preserve_range=True, multichannel=True)
        
        # Convert blurred image back to the original data type.
        blurred = blurred.astype(image.dtype)

        # Merge the unblurred original image into the blurred padded image.
        if image.ndim == 3:
            blurred[pad_top:pad_top+h, pad_left:pad_left+w, :] = image
        else:
            blurred[pad_top:pad_top+h, pad_left:pad_left+w] = image
        return blurred
    return padded

# @tf.function
# def tf_pad_and_resize(image, target_dim):
#     """
#     Pads an input image to be square using the largest image dimension,
#     then resizes it to a square image with dimensions (target_dim x target_dim).

#     Args:
#         image: A TensorFlow tensor representing the input image (grayscale or color).
#         target_dim: An integer; the desired output dimension (target_dim x target_dim).
#         mask_value: An integer; the value to use for padding. Although the original code
#                     uses this to determine a constant fill value, here we replicate the edge
#                     using "SYMMETRIC" padding.

#     Returns:
#         A TensorFlow tensor representing the padded and resized image. If the input image
#         is of type tf.uint8, the output will also be tf.uint8; otherwise it remains in float.
#     """
#     # For uint8 images, convert to float32 [0,1] for processing.
#     if image.dtype == tf.uint8:
#         image_float = tf.cast(image, tf.float32) / 255.0
#     else:
#         image_float = image

#     # Get image dimensions.
#     shape = tf.shape(image_float)
#     h = shape[0]
#     w = shape[1]
#     max_side = tf.maximum(h, w)

#     # Calculate required padding.
#     pad_height = max_side - h
#     pad_width = max_side - w
#     pad_top = tf.math.floordiv(pad_height, 2)
#     pad_bottom = pad_height - pad_top
#     pad_left = tf.math.floordiv(pad_width, 2)
#     pad_right = pad_width - pad_left

#     # Prepare the padding configuration depending on image rank.
#     # Use dynamic rank to determine padding dimensions.
#     rank = tf.rank(image_float)
    
#     def pad_color():
#         return tf.stack([
#             tf.stack([pad_top, pad_bottom]),
#             tf.stack([pad_left, pad_right]),
#             tf.stack([tf.constant(0, tf.int32), tf.constant(0, tf.int32)])
#         ])

#     def pad_grayscale():
#         return tf.stack([
#             tf.stack([pad_top, pad_bottom]),
#             tf.stack([pad_left, pad_right])
#         ])

#     def pad_unsupported():
#         err = tf.debugging.Assert(False, ["Unsupported image dimensions."])
#         with tf.control_dependencies([err]):
#             # Return a dummy tensor of the expected shape (for color images, 3×2)
#             return tf.zeros([3, 2], dtype=tf.int32)

#     paddings = tf.cond(
#         tf.equal(rank, 3),
#         pad_color,
#         lambda: tf.cond(
#             tf.equal(rank, 2),
#             pad_grayscale,
#             pad_unsupported
#         )
#     )
    
#     # if image_float.shape.ndims == 3:
#     #     paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
#     # elif image_float.shape.ndims == 2:
#     #     paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
#     # else:
#     #     raise ValueError("Unsupported image dimensions.")

#     # Pad the image using symmetric padding (which replicates edge values).
#     padded = tf.pad(image_float, paddings, mode="SYMMETRIC")

#     padded.set_shape([None, None, 3])
#     # Resize the padded image to (target_dim, target_dim) with anti-aliasing.
#     resized = tf.image.resize(padded, [target_dim, target_dim], antialias=False)

#     # If the original image was uint8, scale back to [0,255] and cast accordingly.
#     if image.dtype == tf.uint8:
#         resized = tf.cast(tf.clip_by_value(resized * 255.0, 0, 255), image.dtype)

#     return resized

@tf.function
def tf_pad_and_resize(image, target_dim):
    """
    Assumes all input images are RGB.
    
    Pads a non-square RGB image to square by extending the edge pixels along the shorter axis,
    then resizes it to (target_dim x target_dim).

    Steps:
      1. Convert uint8 images to float32 in [0,1].
      2. Squeeze out any extra dimensions.
      3. Enforce a rank-3 shape ([height, width, 3]).
      4. Depending on whether the image is wider or taller, pad vertically or horizontally
         by repeating the edge row/column.
      5. Ensure a static shape is set so tf.image.resize can operate.
      6. Resize the padded image.
      7. If the original image was uint8, convert back to that range.
      
    Args:
      image: A tensor representing the image. Expected shape is [H, W, 3] or with extra singleton dims.
      target_dim: The integer size of the output square image.
      mask_value: Unused here.

    Returns:
      A resized image tensor, with the same dtype as the input.
    """
    orig_dtype = image.dtype
    # Convert to float32 in [0, 1] if needed.
    if orig_dtype == tf.uint8:
        image = tf.cast(image, tf.float32) / 255.0

    # Remove extra singleton dimensions (e.g. a batch dimension of 1).
    image = tf.squeeze(image)
    # Ensure the image is treated as RGB (rank 3).
    image = tf.ensure_shape(image, [None, None, 3])
    
    # Get dynamic height and width.
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    
    # Define functions to pad along the vertical axis (if h < w) or horizontal axis (if w < h).
    def pad_vertical():
        pad_total = w - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        top_pad = tf.repeat(tf.expand_dims(image[0, :, :], axis=0), pad_top, axis=0)
        bottom_pad = tf.repeat(tf.expand_dims(image[-1, :, :], axis=0), pad_bottom, axis=0)
        return tf.concat([top_pad, image, bottom_pad], axis=0)
    
    def pad_horizontal():
        pad_total = h - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        left_pad = tf.repeat(tf.expand_dims(image[:, 0, :], axis=1), pad_left, axis=1)
        right_pad = tf.repeat(tf.expand_dims(image[:, -1, :], axis=1), pad_right, axis=1)
        return tf.concat([left_pad, image, right_pad], axis=1)
    
    # Choose the appropriate padding branch.
    padded = tf.cond(tf.less(h, w),
                     pad_vertical,
                     lambda: tf.cond(tf.less(w, h),
                                     pad_horizontal,
                                     lambda: image))
    
    # Inform TensorFlow that the padded image is RGB.
    padded = tf.ensure_shape(padded, [None, None, 3])
    
    # Resize the padded image.
    resized = tf.image.resize(padded, [target_dim, target_dim], antialias=False)
    
    # Convert back to the original dtype if necessary.
    if orig_dtype == tf.uint8:
        resized = tf.cast(tf.clip_by_value(resized * 255.0, 0, 255), orig_dtype)
    
    return resized


def rescale_image(image: np.ndarray, config: Config) -> np.ndarray:
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

@tf.function
def tf_rescale_image(image, config: Config):
    """
    Rescales an image's pixel values to a target range defined in config.RESCALE_PIXELS.
    Also, if the image's dtype differs from config.IMAGE_PVALUE_TYPE, it is converted.

    Args:
        image: A TensorFlow tensor representing the input image.
        config: A configuration object containing:
            - RESCALE_PIXELS: A tuple/list with two numbers (target_min, target_max).
            - IMAGE_PVALUE_TYPE: The desired tf.DType for the image (e.g., tf.uint8 or tf.float32).

    Returns:
        A TensorFlow tensor: The rescaled image with pixel values in the target range and of type config.IMAGE_PVALUE_TYPE.
    """
    # Ensure image is in the desired type.
    if image.dtype != config.IMAGE_PVALUE_TYPE:
        image = tf.cast(image, config.IMAGE_PVALUE_TYPE)
    
    # Use float32 for rescaling computations.
    image_float = tf.cast(image, tf.float32)
    
    # Determine input range by checking the maximum pixel value.
    max_val = tf.reduce_max(image_float)
    input_min, input_max = tf.cond(
        tf.less_equal(max_val, 1.0),
        lambda: (tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32)),
        lambda: (tf.constant(0.0, tf.float32), tf.constant(255.0, tf.float32))
    )
    
    # Get target range from config.
    target_min = tf.constant(config.RESCALE_PIXELS[0], tf.float32)
    target_max = tf.constant(config.RESCALE_PIXELS[1], tf.float32)
    
    # Clip image values to the detected input range.
    image_clipped = tf.clip_by_value(image_float, input_min, input_max)
    
    # Avoid division by zero. (In our case input_max != input_min, but this is a safe guard.)
    scaled_image = tf.cond(
        tf.equal(input_max, input_min),
        lambda: tf.fill(tf.shape(image_clipped), target_min),
        lambda: ((image_clipped - input_min) / (input_max - input_min)) * (target_max - target_min) + target_min
    )
    
    # If the target type is integer, round the values before casting.
    if config.IMAGE_PVALUE_TYPE in [tf.uint8, tf.int32, tf.int16]:
        scaled_image = tf.cast(tf.round(scaled_image), config.IMAGE_PVALUE_TYPE)
    else:
        scaled_image = tf.cast(scaled_image, config.IMAGE_PVALUE_TYPE)
    
    return scaled_image

def preprocess_record(
    record1: list,
    config: Config,
) -> Tuple[np.ndarray, str, bool]:
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
    record = []
    for r in record1:
        if isinstance(r, tf.Tensor):
            if r.dtype == tf.string:
                record.append(r.numpy().decode('utf-8'))
            else:
                record.append(r.numpy())
        else:
            record.append(r)
    
    # IMAGE PREPROCESSING
    img_name = record[0]
    img_path = os.path.join(config.DATA_DIR, config.IMAGE_DIR, img_name)
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    image = io.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    
    img_float = image.copy()
    # Convert to float for processing
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0

    if config.NORMALIZE_IMAGES:
        img_float = dental_gray_world_white_balance(img_float)
        #output is float [0-1]
    # Mask
    # if config.MASK_BG:
    img_float = resize_and_mask_background(image = img_float, polygon = record[1:3], config = config)
    
    # Pad and resize
    img_float = pad_image(img_float, target_dim=config.TARGET_DIM, config=config)
    
    #rescale:
    # if config.RESCALE_PIXELS[0] is not None:
    #     image = rescale_image(image, config)

    return img_float, record[3], record[4]

@tf.function
def tf_preprocess_record(record1, config):
    """
    Preprocess a single record in the tooth classification data pipeline.
    This function loads an image specified by the input record, applies various
    preprocessing steps such as white-balancing, background masking, padding/resizing,
    and optional pixel rescaling based on the configuration provided.

    Parameters:
        record1 (list or tuple): Expected to contain:
            - record1[0]: image filename (tf.string)
            - record1[1:3]: tooth polygon/mask (can be tensors or arrays)
            - record1[3]: associated label (e.g. tf.int32)
            - record1[4]: an additional flag (e.g. tf.bool)
        config: A configuration object with attributes like:
            - DATA_DIR (str)
            - IMAGE_DIR (str)
            - NORMALIZE_IMAGES (bool)
            - MASK_BG (bool)
            - TARGET_DIM (int or tuple)
            - MASK_VALUE (numeric)
            - RESCALE_PIXELS (list, where element 0 being not None means rescaling is enabled)
            
    Returns:
        A tuple: (preprocessed image, label, additional_flag)
    """
    # (Assumes record1 elements are already tf.Tensors; no conversion loop is needed)
    img_name = record1[0]  # expected to be a tf.string

    # Build full image path using tf.string.join.
    # Note: os.sep is used as separator; you may also use "/" if that is acceptable.
    img_path = tf.strings.join([config.DATA_DIR, config.IMAGE_DIR, img_name], separator=os.sep)

    

    # Read the image file and decode it.
    image_file = tf.io.read_file(img_path)
    
    # Check if the file exists. Since TF does not have a direct graph equivalent of os.path.exists,
    static_img_path = tf.get_static_value(img_path)
    tf.debugging.assert_greater(
        tf.size(image_file),
        0,
        message="Image file is empty or not found: " + (static_img_path if static_img_path is not None else "unknown")
    )
    
    
    image = tf.image.decode_image(image_file, channels=3)

    # If image normalization is enabled, apply the white-balancing function.
    if config.NORMALIZE_IMAGES:
        image = tf_dental_gray_world_white_balance(image)

    # Apply background masking if enabled.
    if config.MASK_BG:
        # Pass the polygon data (record1[1:3]) along with the image and config.
        image = tf_mask_background(image, record1[1:3], config)

    # Pad and resize the image.
    image = tf_pad_and_resize(image, target_dim=config.TARGET_DIM)

    # Optionally rescale the image pixels.
    if config.RESCALE_PIXELS[0] is not None:
        image = tf_rescale_image(image, config)

    # Return the processed image along with label and additional flag.
    return image, record1[3], record1[4]

def preprocess_raw_dataset(
    json_annotations_path: str,
    output_dir: str,
    config: Config,
    copy_images: bool = False,
    verbose: bool = False,
) -> None:
    """
    Goes through each annotation, optionally remove loads image, removes dark images from the dataset, and saves the final json and images to output_dir 
    with the appropriate label in the filename or subfolder.
    This function is for manual preprocessing 
    
    Note: This function does not perform oversampling and augmentation as the preprocess_record() does.

    """
    os.makedirs(output_dir, exist_ok=True)
    
    #loading the raw json file
    with open(json_annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f) 
    
    #filtering out the dark images
    image_dir = os.path.join(config.DATA_DIR, config.IMAGE_DIR)
    if config.REMOVE_DARK_IMAGES:
        if verbose:
            print("Filtering out dark images...")
            
        annotations, removed_images_list = remove_dark_images_from_json(annotations, image_dir,config.DARK_IMAGE_THRESHOLD )
        if verbose:
            print(f"Number of removed images annotations: {len(removed_images_list)}")
        
        # Get the current datetime and format it so it's safe to use in a filename.
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"removed_dark_images_{current_datetime}.txt"
        log_path = os.path.join(output_dir,filename)

        # Open the file in write mode and write the strings and list items.
        if verbose:
            print(f"Saving log of dark image removal at: {os.path.join(output_dir,filename)}")
        with open(log_path, "w", encoding = "utf-8") as file:
            file.write(f"source: {json_annotations_path}\n")
            file.write(f"darkness threshold: {config.DARK_IMAGE_THRESHOLD}\n")
            file.write(f"removed images count: {len(removed_images_list)}\n")
            file.write("Image names:\n\n")
            
            for image in removed_images_list:
                file.write(f"{image}\n")
    
    #converting the raw json file to the format required for the target class
    if verbose:
        print("Converting annotations to toothwise json file.")
    annotations = convert_annotations(annotations, config.TARGET_CLASS)
    if verbose:
        print(f"Number of final annotations: {len(annotations)}")

    annotations = convert_int32_to_int(annotations)
    f_annot_path = os.path.join(output_dir,f"filtered_{config.TARGET_CLASS}_annotations.json")
    if verbose: 
        print(f"Saving filtered annotations to {f_annot_path}")
    with open(f_annot_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    
    #initiating a progress bar
    if copy_images:
        if verbose:
            print(f"Copying images to: {output_dir}")
        progress_bar = tqdm(total=len(annotations), desc="Copying images", unit="image")
        
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
            # Construct output filename
            out_path = os.path.join(output_dir, img_name)
            
            io.imsave(out_path, image)
            # print(f"Saved: {out_path}")
            
            progress_bar.update(1)
        
        progress_bar.close()
    if verbose: 
        print("Done pre-processing the dataset.")
    
def preprocess_dataset_manual(
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
    
    Note: This function does not perform oversampling and augmentation as the preprocess_record() does.

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
                        config: Config,
                        verbose: bool = True) -> None:
    """
    Loads the new-master dataset JSON (converted from the raw json) and splits it into train, validation, and test sets stratified by target_class.
    Writes three JSON files to output_dir with the same structure as the master JSON.
    This function needs to be run manually. This helps make sure about test set separation during the experiment.
    
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
    train_ratio = config.TRAIN_VAL_TEST_RATIOS[0]
    val_ratio = config.TRAIN_VAL_TEST_RATIOS[1]
    test_ratio = config.TRAIN_VAL_TEST_RATIOS[2]
    random_seed = config.RANDOM_SEED

    # Ensure the ratios sum up to 1
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum up to 1.")
    
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
    path = os.path.join(output_dir, f"{Config.TARGET_CLASS}_filtered_train.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=2)
    if verbose:
        print(f"JSON file save at: {path}")
        
    path = os.path.join(output_dir, f"{Config.TARGET_CLASS}_filtered_val.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(val_json, f, indent=2)
    if verbose:
        print(f"JSON file save at: {path}")
        
    path = os.path.join(output_dir, f"{Config.TARGET_CLASS}_filtered_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, indent=2)
    if verbose:
        print(f"JSON file save at: {path}")


def build_tf_dataset(
    records: list,
    config: Config,
) -> tf.data.Dataset:
    
    # the rotation layer for augmentation needs to be created outside of the prep function and then accessed there
    random_rotation_layer = tf.keras.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode= "nearest")
    
    def _load_and_preprocess(*record_parts):
        # Call preprocess_record via tf.py_function. It returns a tuple:
        # (preprocessed image, label, augmentation flag).
        record = tuple(record_parts)
        image, label, aug = tf.py_function(
            func=lambda a, b, c, d, e: preprocess_record([a, b, c, d, e], config),
            inp=record,
            Tout=(tf.float32, tf.string, tf.bool)
        )
        
        # Set static shapes if you know the expected dimensions.
        image.set_shape([config.INPUT_SHAPE[0], config.INPUT_SHAPE[1], config.INPUT_SHAPE[2]])
        label.set_shape([])
        aug.set_shape([])
        
        # the function for augmentation
        # random_rotation_layer = tf.keras.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode= "nearest")
        
        def augment_fn(img):
            img = tf.image.random_flip_left_right(img)
            img = random_rotation_layer(img)
            # img = tf.image.random_brightness(img, max_delta=0.1)
            return img
        
        # If the augmentation flag is True, apply additional augmentations.
        image = tf.cond(aug, lambda: augment_fn(image), lambda: image)
        
        # Convert the label to an integer class: if it matches config.TARGET_CLASS, output 1; else 0.
        label_int = tf.cond(
            tf.equal(label, tf.constant(config.TARGET_CLASS, dtype=tf.string)),
            lambda: tf.constant(1., dtype=tf.float32),
            lambda: tf.constant(0., dtype=tf.float32)
        )
        
        return image, label_int
    
    # Create a dataset from the list of records.
    def record_generator():
        for record in records:
            # print(f"yeilding record:{record[0]}")
            yield tuple(record)

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.bool)
    )

    ds = tf.data.Dataset.from_generator(
        record_generator,
        output_signature=output_signature
    )

    
    # Map the _load_and_preprocess function in parallel.
    ds = ds.map(_load_and_preprocess, num_parallel_calls= tf.data.AUTOTUNE)
    ds = ds.cache()
    # Optionally shuffle the dataset.
    if config.SHUFFLE_DATASET:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)#len(records), reshuffle_each_iteration=True)
    
    # Batch the data and prefetch for optimal pipeline performance.
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

def tf_build_tf_dataset(records: list, config: Config) -> tf.data.Dataset:
    # Create a rotation layer for augmentation outside the preprocessing function.
    random_rotation_layer = tf.keras.layers.RandomRotation(
        factor=(-0.1, 0.1), fill_mode="nearest"
    )
    
    # if config.MASK_VALUE in range(5,50):
    #     init_cached_gaussian_kernel(config)
    
    def _load_and_preprocess(*record_parts):
        # Convert record parts to a list.
        record = list(record_parts)
        # Directly call the TensorFlow version of preprocess_record.
        image, label, aug = tf_preprocess_record(record, config)
        
        # Set static shapes if known.
        image.set_shape([config.INPUT_SHAPE[0], config.INPUT_SHAPE[1], config.INPUT_SHAPE[2]])
        label.set_shape([])
        aug.set_shape([])
        
        # Define augmentation function.
        def augment_fn(img):
            img = tf.image.random_flip_left_right(img)
            img = random_rotation_layer(img)
            return img
        
        # Apply augmentation conditionally.
        image = tf.cond(aug, lambda: augment_fn(image), lambda: image)
        
        # Convert label: if it matches TARGET_CLASS, output 1; else 0.
        label_int = tf.cond(
            tf.equal(label, tf.constant(config.TARGET_CLASS, dtype=tf.string)),
            lambda: tf.constant(1., dtype=tf.float32),
            lambda: tf.constant(0., dtype=tf.float32)
        )
        
        return image, label_int

    # Generator function yielding records.
    def record_generator():
        for record in records:
            yield tuple(record)

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.bool)
    )

    # Build dataset from generator.
    ds = tf.data.Dataset.from_generator(
        record_generator,
        output_signature=output_signature
    )
    
    # Map the _load_and_preprocess function in parallel.
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache()
    ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
    # Optionally shuffle the dataset.
    if config.SHUFFLE_DATASET:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    
    # Batch and prefetch the data.
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

# --- Revised Dataset Builder ---
def tf_build_tf_dataset_optimized(
    all_records: list, # Takes the direct output of parse_dataset_json
    config: Config
) -> tf.data.Dataset:
    """
    Builds an optimized tf.data.Dataset using from_tensor_slices,
    handling ragged tensors for polygon data.

    Args:
        all_records: List of records, output from parse_dataset_json.
        config: Configuration object.

    Returns:
        An optimized tf.data.Dataset.
    """
    # 1. Prepare data for from_tensor_slices
    full_image_paths = [r[0] for r in all_records]
    polygon_xs = [r[1] for r in all_records] # List of lists (ragged)
    polygon_ys = [r[2] for r in all_records] # List of lists (ragged)
    string_labels = [r[3] for r in all_records]
    augment_flags = [r[4] for r in all_records]

    # Create TensorFlow constants, using RaggedTensor for polygons
    paths_tensor = tf.constant(full_image_paths)
    # Use from_generator if creating ragged constant directly is problematic
    # or tf.ragged.constant if lists are compatible
    poly_x_tensor = tf.ragged.constant(polygon_xs, dtype=tf.float32) # Assuming coords are floats
    poly_y_tensor = tf.ragged.constant(polygon_ys, dtype=tf.float32) # Assuming coords are floats
    labels_tensor = tf.constant(string_labels)
    flags_tensor = tf.constant(augment_flags)


    # 2. Create the initial dataset from slices
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths_tensor, poly_x_tensor, poly_y_tensor, labels_tensor, flags_tensor)
    )

    # 3. Define preprocessing and augmentation function to map
    # Create rotation layer outside the map function.
    random_rotation_layer = tf.keras.layers.RandomRotation(
        factor=(-0.1, 0.1), fill_mode="nearest"
    )

    def _load_preprocess_augment(path, poly_x, poly_y, label, aug_flag):
        # Create the record1 input as expected by tf_preprocess_record
        record1 = (path, poly_x, poly_y, label, aug_flag)
        processed_image, processed_label, processed_flag = tf_preprocess_record(record1, config)

        # Set static shape for the processed image
        processed_image.set_shape([config.INPUT_SHAPE[0], config.INPUT_SHAPE[1], config.INPUT_SHAPE[2]])

        # Define augmentation function (applied to the processed image).
        def augment_fn(img):
            img = tf.image.random_flip_left_right(img)
            img = random_rotation_layer(img)
            return img

        # Apply augmentation conditionally using the original flag from the dataset slice.
        final_image = tf.cond(aug_flag, lambda: augment_fn(processed_image), lambda: processed_image)

        # Convert label: if it matches TARGET_CLASS, output 1; else 0.
        final_label = tf.cond(
            tf.equal(label, tf.constant(config.TARGET_CLASS, dtype=tf.string)),
            lambda: tf.constant(1., dtype=tf.float32),
            lambda: tf.constant(0., dtype=tf.float32)
        )
        final_label.set_shape([]) # Set shape for the final scalar label

        return final_image, final_label # Return only the final image and label needed for training

    # 4. Map the function
    ds = dataset.map(_load_preprocess_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Apply optimizations (Caching, Shuffling, Batching, Prefetching)

    # Cache after mapping if dataset fits in memory
    # ds = ds.cache() # Uncomment if needed and feasible

    # Shuffle after caching (if used)
    if config.SHUFFLE_DATASET:
        # Adjust buffer size based on memory, len(all_records) is a good default
        ds = ds.shuffle(buffer_size=len(all_records), reshuffle_each_iteration=True)

    # Batch the dataset
    ds = ds.batch(config.BATCH_SIZE)

    # Prefetch to overlap CPU preprocessing and GPU training
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Remove explicit copy_to_device unless profiling shows it's beneficial
    # ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))

    return ds

def preprocess_and_save_images(all_records: list, config: Config, set_name: str) -> list:
    """
    Preprocess each record and save the resulting image to disk only once.
    Records that are augmentation copies (aug_flag==True) point to the same saved file.
    The output folder is built using a code derived from several config variables:
        TARGET_CLASS, MASK_VALUE, RANDOM_SEED, AUGMENT_DATA, NORMALIZE_IMAGES,
        MASK_BG, DARK_IMAGE_THRESHOLD, POLYGON_SMOOTHING_TOLERANCE.
    
    If the folder (and records file) already exists, the function loads the saved records list.
    
    Each record is expected to be a list:
        [original_image_name, polygon_x, polygon_y, label, augmentation_flag]
    """
    # Build a folder code using the configuration parameters.
    folder_code = (
        f"{config.TARGET_CLASS}_"
        f"{config.MASK_VALUE}_"
        f"{config.RANDOM_SEED}_"
        f"{int(config.AUGMENT_DATA)}_"  # Convert bool to 0 or 1.
        f"{int(config.NORMALIZE_IMAGES)}_"
        f"{int(config.MASK_BG)}_"
        f"{config.DARK_IMAGE_THRESHOLD}_"
        f"{config.POLYGON_SMOOTHING_TOLERANCE}"
    )
    
    processed_folder = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, folder_code)
    records_file = os.path.join(processed_folder, f"{set_name}_records.json")
    
    # If the folder and records file already exist, load and return the records.
    if os.path.exists(processed_folder) and os.path.exists(records_file):
        print(f"Processed folder '{processed_folder}' and records file exist. Loading records...")
        with open(records_file, "r") as f:
            updated_records = json.load(f)
        return updated_records, folder_code
    
    # If the folder exists but records file doesn't, or if the folder doesn't exist, process the images.
    os.makedirs(processed_folder, exist_ok=True)
    processed_images = {}  # Track images already processed by base name.
    updated_records = []

    for record in all_records:
        original_path, poly_x, poly_y, label, aug_flag = record
        base_name = os.path.basename(original_path)
        
        # For augmentation copies, use the same preprocessed file as the original.
        if base_name in processed_images:
            new_path = processed_images[base_name]
        else:
            # Process the image using your existing preprocess_record function.
            processed_image, processed_label, processed_aug_flag = preprocess_record(record, config)
            new_path = os.path.join(processed_folder, base_name)
            # Convert image to uint8 if necessary.
            if processed_image.dtype != np.uint8 and processed_image.max() < 2:
                img_to_save = (processed_image * 255).astype(np.uint8)
            else:
                img_to_save = processed_image.astype(np.uint8)
            imsave(new_path, img_to_save)
            processed_images[base_name] = new_path

        # Update the record to point to the new preprocessed image path.
        updated_record = [new_path, poly_x, poly_y, label, aug_flag]
        updated_records.append(updated_record)

    # Save the updated records list for future runs.
    with open(records_file, "w") as f:
        json.dump(updated_records, f)
    print(f"Preprocessing complete. Records saved to {records_file}")
    
    return updated_records, folder_code

def build_tf_dataset_from_preprocessed(records: list, config) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset from preprocessed images saved on disk.
    Each record is assumed to be in the format:
        [image_path, polygon_x, polygon_y, label, augmentation_flag]
    
    Images are loaded from disk and, if the augmentation flag is True,
    the image is augmented on the fly.
    """
    # Unpack fields from records.
    image_paths = [r[0] for r in records]
    polygon_xs = [r[1] for r in records]  # if needed later
    polygon_ys = [r[2] for r in records]  # if needed later
    labels = [r[3] for r in records]
    aug_flags = [r[4] for r in records]

    # Create TensorFlow tensors.
    paths_tensor = tf.constant(image_paths)
    poly_x_tensor = tf.ragged.constant(polygon_xs, dtype=tf.float32)
    poly_y_tensor = tf.ragged.constant(polygon_ys, dtype=tf.float32)
    labels_tensor = tf.constant(labels)
    flags_tensor = tf.constant(aug_flags)

    # Create the dataset.
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths_tensor, poly_x_tensor, poly_y_tensor, labels_tensor, flags_tensor)
    )

    # Create an augmentation layer, for example, a random rotation layer.
    random_rotation_layer = tf.keras.layers.RandomRotation(
        factor=(-0.1, 0.1), fill_mode="nearest"
    )

    def _load_image(path, poly_x, poly_y, label, aug_flag):
        # Load the preprocessed image from disk.
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        # Resize image to the desired input dimensions if needed.
        image = tf.image.resize(image, [config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]])
        
        # Apply augmentation conditionally.
        def augment_fn(img):
            img = tf.image.random_flip_left_right(img)
            img = random_rotation_layer(img)
            return img

        final_image = tf.cond(aug_flag, lambda: augment_fn(image), lambda: image)
        
        # Convert label: if it matches TARGET_CLASS, output 1; else 0.
        final_label = tf.cond(
            tf.equal(label, tf.constant(config.TARGET_CLASS, dtype=tf.string)),
            lambda: tf.constant(1.0, dtype=tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        final_label.set_shape([])
        return final_image, final_label

    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if config.SHUFFLE_DATASET:
        dataset = dataset.shuffle(buffer_size=len(records), reshuffle_each_iteration=True)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    """
    If this file is run as a script (e.g., python data_pipeline.py),
    you might put a small demo or test code here.
    """
    # print(f"yeilding record:{record[0]}")
    pass
