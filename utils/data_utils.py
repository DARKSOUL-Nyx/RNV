import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm # A progress bar library, run: pip install tqdm
from .mask_utils import generate_mask_from_json # Add this import

def get_image_ids(images_dir):
    """Scans the images directory and returns a list of unique image IDs."""
    image_files = os.listdir(images_dir)
    image_ids = list(set([f.split('_pre_disaster.png')[0] for f in image_files if f.endswith('_pre_disaster.png')]))
    return sorted(image_ids)

def create_stratified_data_splits(image_ids, labels_dir, test_size=0.2, random_state=42):
    """
    Scans target masks to find which images have damage and performs a stratified split.
    This ensures that the train and validation sets have a similar distribution of damaged/undamaged examples.
    """
    print("Scanning masks for stratification...")
    labels = []
    for img_id in tqdm(image_ids):
        post_label_path = os.path.join(labels_dir, f"{img_id}_post_disaster.json")
        # Generate the mask to check for damage
        mask_array = generate_mask_from_json(post_label_path)
        has_damage = 1 if np.max(mask_array) > 1 else 0
        labels.append(has_damage)

    
    # Check the balance of the dataset
    print(f"Dataset balance: {np.sum(labels)} images with damage out of {len(labels)} total.")
    
    # Perform the stratified split
    train_ids, val_ids, _, _ = train_test_split(
        image_ids, 
        labels,
        test_size=test_size, 
        random_state=random_state,
        stratify=labels # This is the key argument for stratification
    )
    
    return train_ids, val_ids