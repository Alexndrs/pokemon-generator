import os
import torch
import torchvision.utils as vutils
from backend.model.unet import UNet
from backend.model.diffusion import GaussianDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples(model_ckpt, output_dir="./samples", num_images=4):
    os.makedirs(output_dir, exist_ok=True)

    # === Load model ===
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=256).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # === Diffusion interface ===
    diffusion = GaussianDiffusion(model=model, device=device, H=128, W=128)

    # === Sampling ===
    with torch.no_grad():
        images = diffusion.sample(batch_size=num_images)  # (B, 3, H, W)

    # === Save grid ===
    grid = vutils.make_grid(images, nrow=4, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, os.path.join(output_dir, "generated_samples.png"))
    print(f"Saved samples to {os.path.join(output_dir, 'generated_samples.png')}")

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "model_epoch_9.pt")
    sample_output_dir = os.path.join(ROOT_DIR, "samples")
    generate_samples(model_ckpt=ckpt_path, output_dir=sample_output_dir)
