import os
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.cuda.amp as amp # For mixed precision training -> improving training
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.data.preprocessing import PokemonDataset
from backend.model.unet import UNet
from backend.model.diffusion import GaussianDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # === Hyperparameters ===
    image_size = 128
    batch_size = 16
    lr = 1e-4
    epochs = 500 # Total number of epochs to train for

    # Save path for checkpoints
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(save_path, exist_ok=True)
    
    # Checkpoint filename
    checkpoint_file = os.path.join(save_path, "last_checkpoint.pt")

    # === Dataset & Dataloader ===
    csv_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv")
    dataset = PokemonDataset(
        csv_path=csv_path,
        image_size=image_size,
        image_type_filter="official",  
        use_metadata=False,
        use_descriptions=False
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # === Model & Diffusion ===
    model = UNet().to(device)
    diffusion = GaussianDiffusion(model=model, device=device, H=image_size, W=image_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize scaler for mixed precision if uncommented
    # scaler = torch.cuda.amp.GradScaler()

    # --- Check for existing checkpoint ---
    start_epoch = 0
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # if 'scaler_state_dict' in checkpoint and 'scaler' in locals():
        #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # === Training loop ===
    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            x_0 = batch["image"].to(device)  # (B, 3, H, W)
            t = torch.randint(0, diffusion.timesteps, (x_0.size(0),), device=device)

            optimizer.zero_grad()
            loss = diffusion.loss(x_0, t)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save checkpoint after each epoch
        # You can adjust the frequency of saving, e.g., every 5 epochs
        # or save only the best model based on validation loss if you add validation
        print(f"Saving checkpoint to {checkpoint_file}...")
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        # if 'scaler' in locals():
        #     checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint_data, checkpoint_file)

    print("Training complete!")

if __name__ == "__main__":
    train()