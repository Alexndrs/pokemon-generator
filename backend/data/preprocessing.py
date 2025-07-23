import os
import json
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PokemonDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 128,
        image_type_filter: Optional[str] = None,  # "official", "game", or None for all
        use_metadata: bool = False,
        use_descriptions: bool = False,
        metadata_path: Optional[str] = None,
        description_path: Optional[str] = None,
    ):
        self.data = pd.read_csv(csv_path)
        
        # Optional filtering (e.g., keep only official images for now)
        if image_type_filter:
            self.data = self.data[self.data["image_type"] == image_type_filter]

        self.image_size = image_size
        self.use_metadata = use_metadata
        self.use_descriptions = use_descriptions

        # Optional metadata loading
        self.metadata = {}
        if use_metadata and metadata_path:
            with open(metadata_path, "r") as f:
                meta_list = json.load(f)
                self.metadata = {m["name"]: m for m in meta_list}

        # Optional description loading
        self.descriptions = {}
        if use_descriptions and description_path:
            with open(description_path, "r") as f:
                desc_list = json.load(f)
                self.descriptions = {d["name"]: d for d in desc_list}

        # Define transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        image_path = row["image_path"].replace("\\", os.sep)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        sample = {
            "image": image,
            "name": row["name"],
            "color": row.get("color", "unknown"),
            "generation_id": row.get("generation_id", 0),
            "primary_type": row.get("primary_type", "unknown"),
            "is_legendary": bool(row.get("is_legendary", False)),
            "is_mythical": bool(row.get("is_mythical", False)),
            "height": row.get("height", 0.0),
            "weight": row.get("weight", 0.0),
            "is_sprite": bool(row.get("is_sprite", True))
        }

        if self.use_metadata:
            sample["metadata"] = self.metadata.get(row["name"], {})

        if self.use_descriptions:
            sample["description"] = self.descriptions.get(row["name"], {}).get("simple_description", "")

        return sample

