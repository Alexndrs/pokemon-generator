import os
import json
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import time
import random


class PokemonDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 128,
        is_sprite: Optional[bool] = None,  # True, False, or None for all
        use_metadata: bool = False,
        use_descriptions: bool = False,
        metadata_path: Optional[str] = None,
        description_path: Optional[str] = None,
        dropout_prob: float = 0.15,
    ):
        self.data = pd.read_csv(csv_path)
        
        # filtering (e.g., keep only official images for now)
        if is_sprite is not None:
            self.data = self.data[self.data["is_sprite"] == is_sprite]


        self.image_size = image_size
        self.use_metadata = use_metadata
        self.use_descriptions = use_descriptions
        self.dropout_prob = dropout_prob

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

        # Define transform image pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.all_types = sorted(self.data["primary_type"].dropna().unique())
        self.all_colors = sorted(self.data["color"].dropna().unique())

        self.type_to_idx = {t: i for i, t in enumerate(self.all_types)}
        self.color_to_idx = {c: i for i, c in enumerate(self.all_colors)}

        # calculate means for missing or default values
        self.mean_height = float(self.data["height"].fillna(0).mean())
        self.mean_weight = float(self.data["weight"].fillna(0).mean())
        self.mean_generation = float(self.data["generation_id"].fillna(0).mean())

    def __len__(self):
        return len(self.data)
    
    def _get_random_seed(self):
        '''Generate a random seed based on the current time (to ensure different seeds at each epoch for different columns droping)'''
        return int(time.time() * 1000000) % 2**32

    
    def encode_with_dropout(self, sample: Dict[str, Any]) -> torch.Tensor:
        '''
        Encode a sample into a feature vector (used for training) with dropout applied to the features
        '''
        
        random.seed(self._get_random_seed())

        # dropout features based on the dropout probability
        drop_type = random.random() < self.dropout_prob
        drop_color = random.random() < self.dropout_prob  
        drop_height = random.random() < self.dropout_prob
        drop_weight = random.random() < self.dropout_prob
        drop_legendary = random.random() < self.dropout_prob
        drop_mythical = random.random() < self.dropout_prob
        drop_generation = random.random() < self.dropout_prob
        drop_sprite = random.random() < self.dropout_prob

        # full dropout with a given proba (classifier free guidance is usually set at 15-20%)
        if random.random() < 0.15:
            return self._encode_empty()

        # One hot encoding for primary type and color
        type_vector = torch.zeros(len(self.all_types))
        color_vector = torch.zeros(len(self.all_colors))

        primary_type = sample.get("primary_type", "unknown") if not drop_type else None
        if primary_type and primary_type in self.type_to_idx:
            type_vector[self.type_to_idx[primary_type]] = 1.0

        color = sample.get("color", "unknown") if not drop_color else None
        if color and color in self.color_to_idx:
            color_vector[self.color_to_idx[color]] = 1.0


        # Numeric features
        height = sample.get("height", 0.0) if not drop_height else self.mean_height
        weight = sample.get("weight", 0.0) if not drop_weight else self.mean_weight
        generation_id = sample.get("generation_id", 0) if not drop_generation else self.mean_generation


        # Binary features
        is_sprite_val = 1.0 if sample.get("is_sprite", True) else 0.0
        if drop_sprite:
            is_sprite_val = 0.5

        is_legendary_val = 1.0 if sample.get("is_legendary", False) else 0.0
        if drop_legendary:
            is_legendary_val = 0.5

        is_mythical_val = 1.0 if sample.get("is_mythical", False) else 0.0
        if drop_mythical:
            is_mythical_val = 0.5

        
        condition_mask = torch.tensor([
            1.0 if not drop_type and primary_type and primary_type != "unknown" else 0.0,
            1.0 if not drop_color and color and color != "unknown" else 0.0,
            1.0 if not drop_height else 0.0,
            1.0 if not drop_weight else 0.0,
            1.0 if not drop_sprite else 0.0,
            1.0 if not drop_legendary else 0.0,
            1.0 if not drop_mythical else 0.0,
            1.0 if not drop_generation else 0.0,
        ], dtype=torch.float32)


        encoding = torch.cat([
            type_vector,
            color_vector,
            torch.tensor([height], dtype=torch.float32),
            torch.tensor([weight], dtype=torch.float32),
            torch.tensor([is_sprite_val], dtype=torch.float32),
            torch.tensor([is_legendary_val], dtype=torch.float32),
            torch.tensor([is_mythical_val], dtype=torch.float32),
            torch.tensor([generation_id], dtype=torch.float32),
            condition_mask
        ])

        return encoding
    
    def _encode_empty(self) -> torch.Tensor:
        '''
        Encode an empty sample (used for classifier free guidance)
        '''
        type_vector = torch.zeros(len(self.all_types))
        color_vector = torch.zeros(len(self.all_colors))
        neutral_vals = torch.tensor([
            self.mean_height,     # height
            self.mean_weight,     # weight  
            0.5,                  # is_sprite
            0.5,                  # is_legendary
            0.5,                  # is_mythical
            self.mean_generation  # generation_id
        ], dtype=torch.float32)

        condition_mask = torch.zeros(8, dtype=torch.float32)
        encoding = torch.cat([
            type_vector,
            color_vector,
            neutral_vals,
            condition_mask
        ])

        return encoding
    
    def encode_full(self, sample: Dict[str, Any]) -> torch.Tensor:
        
        # One hot encoding for primary type and color
        type_vector = torch.zeros(len(self.all_types))
        color_vector = torch.zeros(len(self.all_colors))

        primary_type = sample.get("primary_type", "unknown")
        color = sample.get("color", "unknown")

        if primary_type in self.type_to_idx:
            type_vector[self.type_to_idx[primary_type]] = 1.0

        if color in self.color_to_idx:
            color_vector[self.color_to_idx[color]] = 1.0

        # Numeric features
        height = torch.tensor([float(sample.get("height", 0.0))], dtype=torch.float32)
        weight = torch.tensor([float(sample.get("weight", 0.0))], dtype=torch.float32)
        generation_id = torch.tensor([float(sample.get("generation_id", 0))], dtype=torch.float32)

        # Binary features
        is_sprite = torch.tensor([1.0 if sample.get("is_sprite", True) else 0.0], dtype=torch.float32)
        is_legendary = torch.tensor([1.0 if sample.get("is_legendary", False) else 0.0], dtype=torch.float32)
        is_mythical = torch.tensor([1.0 if sample.get("is_mythical", False) else 0.0], dtype=torch.float32)

        # Masque complet (toutes les conditions actives)
        condition_mask = torch.ones(8, dtype=torch.float32)

        encoding = torch.cat([
            type_vector,
            color_vector,
            height,
            weight,
            is_sprite,
            is_legendary,
            is_mythical,
            generation_id,
            condition_mask
        ])

        return encoding


    def encode_user_request(
        self,
        primary_type: Optional[str] = None,
        color: Optional[str] = None,
        height: Optional[float] = None,
        weight: Optional[float] = None,
        is_legendary: Optional[bool] = None,
        is_mythical: Optional[bool] = None,
        generation_id: Optional[int] = None,
        is_sprite: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Encode a user request into a feature vector (useful for generating new Pokémon)
        """
        type_vector = torch.zeros(len(self.all_types))
        color_vector = torch.zeros(len(self.all_colors))
        if primary_type and primary_type in self.type_to_idx:
            type_vector[self.type_to_idx[primary_type]] = 1.0
        
        if color and color in self.color_to_idx:
            color_vector[self.color_to_idx[color]] = 1.0
        
        height_val = height if height is not None else self.mean_height
        weight_val = weight if weight is not None else self.mean_weight
        gen_val = generation_id if generation_id is not None else self.mean_generation

        sprite_val = (1.0 if is_sprite else 0.0) if is_sprite is not None else 0.5
        legendary_val = (1.0 if is_legendary else 0.0) if is_legendary is not None else 0.5
        mythical_val = (1.0 if is_mythical else 0.0) if is_mythical is not None else 0.5

        condition_mask = torch.tensor([
            1.0 if primary_type is not None else 0.0,
            1.0 if color is not None else 0.0,
            1.0 if height is not None else 0.0,
            1.0 if weight is not None else 0.0,
            1.0 if is_sprite is not None else 0.0,
            1.0 if is_legendary is not None else 0.0,
            1.0 if is_mythical is not None else 0.0,
            1.0 if generation_id is not None else 0.0,
        ], dtype=torch.float32)
        
        encoding = torch.cat([
            type_vector,
            color_vector,
            torch.tensor([height_val], dtype=torch.float32),
            torch.tensor([weight_val], dtype=torch.float32),
            torch.tensor([sprite_val], dtype=torch.float32),
            torch.tensor([legendary_val], dtype=torch.float32),
            torch.tensor([mythical_val], dtype=torch.float32),
            torch.tensor([gen_val], dtype=torch.float32),
            condition_mask
        ])
        
        return encoding
    
    
    def get_request_suffix(
        self,
        primary_type: Optional[str] = None,
        color: Optional[str] = None,
        height: Optional[float] = None,
        weight: Optional[float] = None,
        is_legendary: Optional[bool] = None,
        is_mythical: Optional[bool] = None,
        generation_id: Optional[int] = None,
        is_sprite: Optional[bool] = None
    ) -> str:
        """
        Generate a suffix string based on the user request parameters
        """
        suffix_parts = []
        
        if color is not None:
            suffix_parts.append(f"{color}")
        
        if primary_type is not None:
            suffix_parts.append(f"{primary_type}")
            
        if is_sprite is not None:
            suffix_parts.append("sprite" if is_sprite else "official")
            
        if is_legendary is not None and is_legendary:
            suffix_parts.append("legendary")
            
        if is_mythical is not None and is_mythical:
            suffix_parts.append("mythical")
            
        if generation_id is not None:
            suffix_parts.append(f"gen{generation_id}")
            
        if height is not None:
            suffix_parts.append(f"h{height:.1f}")
            
        if weight is not None:
            suffix_parts.append(f"w{weight:.1f}")
        
        return "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    def get_condition_size(self) -> int:
        '''return conditional vector size'''
        return len(self.all_types) + len(self.all_colors) + 6 + 8  # +6 for other features, +8 for condition mask


    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        image_path = row["image_path"].replace("\\", os.sep)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        sample = {
            "image": image,
            "name": row["name"],
            "color": row.get("color", "unknown"),
            "generation_id": row.get("generation_id", self.mean_generation),
            "primary_type": row.get("primary_type", "unknown"),
            "is_legendary": bool(row.get("is_legendary", False)),
            "is_mythical": bool(row.get("is_mythical", False)),
            "height": row.get("height", self.mean_height),
            "weight": row.get("weight", self.mean_weight),
            "is_sprite": bool(row.get("is_sprite", True))
        }

        if self.use_metadata:
            sample["metadata"] = self.metadata.get(row["name"], {})

        if self.use_descriptions:
            sample["description"] = self.descriptions.get(row["name"], {}).get("simple_description", "")
        
        sample["encoded"] = self.encode_with_dropout(sample)
        sample["full_encoded"] = self.encode_full(sample)

        return sample

