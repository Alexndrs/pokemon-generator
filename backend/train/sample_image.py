import os
import torch
import torchvision.utils as vutils
from backend.model.unet import UNet
from backend.model.diffusion import GaussianDiffusion
from backend.data.preprocessing import PokemonDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(model_ckpt, output_dir="./samples", cond=None, num_images=4, suffix=None):
    os.makedirs(output_dir, exist_ok=True)

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset = PokemonDataset(
        csv_path=os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv"),
        image_size=128,
        is_sprite=None,
        use_metadata=False,
        use_descriptions=False,
    )

    empty_condition_vector = dataset._encode_empty().to(device)

    if cond is None:
        # If no condition is provided, use an empty condition vector
        cond = empty_condition_vector


    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    
    cond_batch = cond.repeat(num_images, 1)  


    # === Load model ===
    model = UNet(empty_condition_vector=empty_condition_vector, in_channels=3, out_channels=3, time_emb_dim=256).to(device)
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # === Diffusion interface ===
    diffusion = GaussianDiffusion(model=model, device=device,empty_condition_vector=empty_condition_vector, H=128, W=128)

    # === Sampling ===
    with torch.no_grad():
        images = diffusion.sample(batch_size=num_images, cond=cond_batch)  # (B, 3, H, W)

    # === Save grid ===
    grid = vutils.make_grid(images, nrow=4, normalize=True, value_range=(-1, 1))
    output_path = os.path.join(output_dir, "generated_samples" + (suffix if suffix else "") + ".png")
    vutils.save_image(grid, output_path)
    print(f"Saved samples to {os.path.join(output_dir, 'generated_samples.png')}")

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "last_checkpoint.pt")
    sample_output_dir = os.path.join(ROOT_DIR, "samples")

    dataset = PokemonDataset(
        csv_path=os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv"),
        image_size=128,
        is_sprite=None,
        use_metadata=False,
        use_descriptions=False,
    )
    conditions = {
        "color": "blue",
        "is_sprite": True
    }


    cond = dataset.encode_user_request(**conditions).to(device)
    suffix = dataset.get_request_suffix(**conditions)

    generate_samples(model_ckpt=ckpt_path, output_dir=sample_output_dir, cond=cond, num_images=4, suffix=suffix)
