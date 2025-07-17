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
    image_type_filter="official",  # or None to include all images
    use_metadata=False,
    use_descriptions=False,
    # metadata_path=meta_path,
    # description_path=desc_path,
)


sample = dataset[0]
print("Pokemon name:", sample["name"])
assert sample is not None, "The sample should not be None"
assert sample["image"].shape == (3, 128, 128), "The image should have shape (3, 128, 128)"
# Show image with matplotlib 
plt.imshow(sample["image"].permute(1, 2, 0))  # (C, H, W) to (H, W, C)
plt.show()


# Testing DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
assert batch["image"].shape == (4, 3, 128, 128), "Batch image shape should be (4, 3, 128, 128)"