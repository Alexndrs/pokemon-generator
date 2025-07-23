# file to execute from root with `python -m backend.data.test_preprocessing` 

import os
from backend.data.preprocessing import PokemonDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv")
desc_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "descriptions.json")
meta_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "metadata.json")




print("Chemin CSV utilisé :", csv_path)

# loading the dataset
dataset = PokemonDataset(
    csv_path=csv_path,
    image_size=128,
    is_sprite=None,
    use_metadata=False,
    use_descriptions=False,
    # metadata_path=meta_path,
    # description_path=desc_path,
)


sample = dataset[0]
print("Pokemon name:", sample["name"])
plt.imshow(sample["image"].permute(1, 2, 0))  # (C, H, W) to (H, W, C)
plt.show()

# Testing DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
print(batch["encoded"].shape)
assert batch["image"].shape == (4, 3, 128, 128), "Batch image shape should be (4, 3, 128, 128)"


empty_condition_vector = dataset._encode_empty()
print("Empty condition vector shape:", empty_condition_vector.shape)