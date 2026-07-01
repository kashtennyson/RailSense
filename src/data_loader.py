import pathlib
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras.applications.resnet50 import preprocess_input
from . import config



def get_train_augmentation():
    """
    Defines the Albumentations pipeline for simulated railway conditions.
    Focuses on environmental noise, lighting, and vibration.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.RandomShadow(p=0.3), 
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
    ])



def apply_augmentation(image):
    """
    Wrapper to apply Albumentations in a TensorFlow map function.
    """
    aug = get_train_augmentation()
    data = {"image": image}
    aug_data = aug(**data)
    return aug_data["image"]



def process_image(file_path, is_training=False):
    """
    Loads, decodes, and optionally augments an image.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SHAPE)
    img = tf.cast(img, tf.float32)

    if is_training:
        # Wrap the numpy-based augmentation for tensorflow
        img = tf.numpy_function(apply_augmentation, [img], tf.float32)
        img.set_shape([*config.IMAGE_SHAPE, 3])

    return img



def prepare_for_autoencoder(image):
    """
    Transforms a single image into a (input, target) pair.
    """
    target = tf.cast(image, tf.float32) / 255.0
    input_tensor = preprocess_input(tf.identity(image))
    
    return input_tensor, target



def load_datasets():
    """
    Loads normal-only datasets for training/validation 
    and a mixed dataset for testing.
    """
    data_root = pathlib.Path(config.DATA_DIR)
    
    # Collect all paths
    all_paths = list(data_root.rglob("*.jpg"))
    
    # Filter paths for normal vs anomaly
    normal_paths = [str(p) for p in all_paths 
                    if "normal" in str(p).lower()]
    anomaly_paths = [str(p) for p in all_paths 
                    if "normal" not in str(p).lower()]
    
    print(f"Found {len(normal_paths)} normal images and {len(anomaly_paths)} anomaly images.")

    # Split normal data for train/val/test
    np.random.seed(config.SEED)
    np.random.shuffle(normal_paths)
    
    n_total = len(normal_paths)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = int(n_total * config.VAL_RATIO)
    
    train_paths = normal_paths[:n_train]
    val_paths = normal_paths[n_train:n_train + n_val]
    test_normal_paths = normal_paths[n_train + n_val:]
    
    test_paths = test_normal_paths + anomaly_paths

    # Create tensorflow dataset
    def create_ds(paths, is_training=False, include_paths=False):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        if is_training:
            ds = ds.shuffle(len(paths))
        
        def _map_fn(path):
            img = process_image(path, is_training)
            input_tensor, target = prepare_for_autoencoder(img)
            if include_paths:
                return input_tensor, target, path
            return input_tensor, target

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = create_ds(train_paths, is_training=True, include_paths=False)
    val_ds = create_ds(val_paths, is_training=False, include_paths=False)
    test_ds = create_ds(test_paths, is_training=False, include_paths=True)

    stats = {
        "n_normal": len(normal_paths),
        "n_anomaly": len(anomaly_paths),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
    }

    return train_ds, val_ds, test_ds, stats