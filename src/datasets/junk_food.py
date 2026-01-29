"""
Dataset classes for loading and preprocessing image data.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class JunkFoodDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        """
        Load junk food dataset from COCO annotations.

        Args:
            data_folder: Path to folder containing images and _annotations.coco.json
            transform: torchvision transforms (optional)
        """
        self.root_dir = data_folder
        self.transform = transform

        # Load annotations
        annotations_path = data_folder + "/_annotations.coco.json"
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create label map from annotations
        label_map = {ann["image_id"]: True for ann in data.get("annotations", [])}

        # Build image list with labels
        self.images = [
            {
                "id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "has_food": img["id"] in label_map,
            }
            for img in data.get("images", [])
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]

        file_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return label as float tensor (0.0 or 1.0) for BCEWithLogitsLoss
        return image, torch.tensor(int(img_info["has_food"]), dtype=torch.float32)
