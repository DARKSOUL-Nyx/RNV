import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Import our new utility
from utils.mask_utils import generate_mask_from_json

class XView2Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_ids, transform=None, target_transform=None):
        """
        Args:
            images_dir (str): Directory with pre/post images.
            labels_dir (str): Directory with the corresponding JSON label files.
            image_ids (list): List of image ids for this dataset split.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir # We now use the labels_dir
        self.image_ids = image_ids
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # --- Load Images (same as before) ---
        pre_image_path = os.path.join(self.images_dir, f"{image_id}_pre_disaster.png")
        post_image_path = os.path.join(self.images_dir, f"{image_id}_post_disaster.png")
        pre_image = Image.open(pre_image_path).convert("RGB")
        post_image = Image.open(post_image_path).convert("RGB")

        # --- Generate Target Mask On-the-Fly ---
        post_label_path = os.path.join(self.labels_dir, f"{image_id}_post_disaster.json")
        target_mask_np = generate_mask_from_json(post_label_path)
        
        # Convert NumPy array to a PIL Image, then to a Tensor
        target_mask = Image.fromarray(target_mask_np)
        target_mask = transforms.ToTensor()(target_mask)

        # --- Apply Transforms ---
        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        
        if self.target_transform:
            target_mask = self.target_transform(target_mask)

        # Create the binary label for ContrastiveLoss (same logic as before)
        has_damage = (torch.max(target_mask) > 1).float() 

        return pre_image, post_image, has_damage