"""
Dataset classes for loading and preprocessing image data.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class JunkFoodBinaryDataset(Dataset):
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


class JunkFoodMulticlassDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        """
        Load junk food dataset for multi-label classification from COCO annotations.
        Excludes the 'junk-food' superclass and creates boolean vectors for specific food items.

        Args:
            data_folder: Path to folder containing images and _annotations.coco.json
            transform: torchvision transforms (optional)
        """
        self.root_dir = data_folder
        self.transform = transform

        # Load annotations
        annotations_path = os.path.join(data_folder, "_annotations.coco.json")
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Filter Categories (Exclude 'junk-food')
        self.categories = [
            cat for cat in data.get("categories", []) if cat["name"] != "junk-food"
        ]
        # Sort by ID to ensure consistent ordering
        self.categories.sort(key=lambda x: x["id"])
        
        # storage for mapping class_id -> index (0..N-1)
        self.cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(self.categories)}
        self.classes = [cat["name"] for cat in self.categories]
        self.num_classes = len(self.classes)

        # 2. Map Image IDs to Annotation Categories
        # Group annotations by image_id, filtering only relevant categories
        img_to_cats = {}
        valid_cat_ids = set(self.cat_id_to_idx.keys())
        
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            
            if cat_id in valid_cat_ids:
                if img_id not in img_to_cats:
                    img_to_cats[img_id] = []
                img_to_cats[img_id].append(cat_id)

        # 3. Build image list
        self.images = []
        for img in data.get("images", []):
            img_id = img["id"]
            
            # Create multi-hot label vector
            label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
            
            # If image has annotations in our valid categories, set them to 1.0
            if img_id in img_to_cats:
                for cat_id in img_to_cats[img_id]:
                    idx = self.cat_id_to_idx[cat_id]
                    label_vector[idx] = 1.0
                    
            self.images.append({
                "file_name": img["file_name"],
                "labels": label_vector
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        file_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return image and multi-hot vector
        return image, img_info["labels"]
