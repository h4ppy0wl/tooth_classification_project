# data_pipeline.py

import os
import glob
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import cv2  # or Pillow (PIL) if you prefer
import skimage
import tensorflow as tf
import random
import json

class InputStream:
    def __init__(self, data):
            self.data = data
            self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


    def access_bit(data1, num):
        """ from bytes array to bits by num position"""
        base = int(num // 8)
        shift = 7 - int(num % 8)
        return (data1[base] & (1 << shift)) >> shift


    def bytes2bit(data):
        """ get bit string from bytes data"""
        string1 = ''.join([str(InputStream.access_bit(data, i)) for i in range(len(data) * 8)])
        return string1


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
    img_float = image_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert to HSV to detect white (teeth) and red (gums/tongue)
    img_hsv = skimage.color.rgb2hsv(img_float)

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

    return skimage.img_as_ubyte(img_float)  # Convert back to uint8

def is_darker_than_threshold(image_path, threshold=0.25):
    image = skimage.io.imread(image_path)
    image_gray = skimage.color.rgb2gray(image)  # Convert to grayscale
    mean_intensity = np.mean(image_gray)
    return mean_intensity < threshold


def load_image_paths(data_dir: str) -> List[str]:
    """
    Recursively find image files (e.g., .jpg, .png) under data_dir.
    Returns a list of file paths.
    """
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    # Also handle other extensions if needed
    # image_paths += glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    return image_paths

def crop_tooth_region(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop an image to a bounding box (x, y, width, height).
    This bounding box presumably comes from a tooth detection/annotation step.
    Returns the cropped tooth region.
    """
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    return cropped

def mask_background(
    image: np.ndarray,
    mask_threshold: int = 230
) -> np.ndarray:
    """
    Example function to mask the background in an image (e.g., near white).
    You would replace this logic with whatever works best for your scenario.
    """
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create mask
    _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
    # Invert mask if necessary, depending on which part you want to keep
    mask_inv = cv2.bitwise_not(mask)
    
    # Expand back to 3 channels
    mask_inv_3c = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    
    # Apply mask
    # everything that is background becomes black
    masked_image = cv2.bitwise_and(image, mask_inv_3c)
    return masked_image

def load_annotations(annotation_file: str) -> pd.DataFrame:
    """
    Reads a CSV or JSON containing bounding boxes, labels (plaque, no plaque, etc.).
    Must match how you store your annotations.
    Example CSV columns could be: [image_path, x, y, w, h, label].
    """
    df = pd.read_csv(annotation_file)  # or pd.read_json, etc.
    return df

def preprocess_and_label(
    df_annotations: pd.DataFrame,
    output_dir: str,
    mask_bg: bool = True
) -> None:
    """
    Goes through each annotation, loads image, crops the tooth region,
    optionally masks the background, and saves the final image to output_dir 
    with the appropriate label in the filename or subfolder.

    Example: output_dir/<label>/image_filename.jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df_annotations.iterrows():
        img_path = row["image_path"]
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        label = row["label"]  # e.g. 'plaque' or 'no_plaque'
        
        if not os.path.exists(img_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Crop
        cropped_img = crop_tooth_region(image, (x, y, w, h))
        
        # Mask
        if mask_bg:
            cropped_img = mask_background(cropped_img)
        
        # Save to <output_dir>/<label> subfolder
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Construct output filename
        base_name = os.path.basename(img_path)
        out_name = f"{os.path.splitext(base_name)[0]}_cropped.jpg"
        out_path = os.path.join(label_dir, out_name)
        
        cv2.imwrite(out_path, cropped_img)

def split_dataset(
    images_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Splits the processed dataset into train, val, test sets.
    Assumes a directory structure like:
        images_dir/plaque/*.jpg
        images_dir/no_plaque/*.jpg
    
    Returns a dictionary with keys: 'train', 'val', 'test', 
    each containing a list of image paths.
    """
    random.seed(random_seed)
    
    all_image_paths = []
    for label in os.listdir(images_dir):
        label_path = os.path.join(images_dir, label)
        if not os.path.isdir(label_path):
            continue
        img_paths = glob.glob(os.path.join(label_path, "*.jpg"))
        
        # Shuffle and split
        random.shuffle(img_paths)
        all_image_paths.extend([(p, label) for p in img_paths])
    
    random.shuffle(all_image_paths)
    total = len(all_image_paths)
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)
    
    train_samples = all_image_paths[:train_count]
    val_samples = all_image_paths[train_count : train_count + val_count]
    test_samples = all_image_paths[train_count + val_count :]
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

def build_tf_dataset(
    samples: List[Tuple[str, str]],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False
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
        image = tf.image.resize(image, (224, 224))  # or any dimension
        image = image / 255.0  # scale to [0, 1]
        
        # Optional: augment
        if augment:
            # simple horizontal flip
            image = tf.image.random_flip_left_right(image)
            # maybe random brightness
            image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Convert label (string) to an integer class if needed
        # For instance, if you have "plaque" -> 1, "no_plaque" -> 0
        label_int = tf.cond(
            tf.math.equal(label, tf.constant("plaque")),
            lambda: tf.constant(1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )
        
        return image, label_int

    # Convert the samples to a TF Dataset
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), reshuffle_each_iteration=True)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

if __name__ == "__main__":
    """
    If this file is run as a script (e.g., python data_pipeline.py),
    you might put a small demo or test code here.
    """
    pass
