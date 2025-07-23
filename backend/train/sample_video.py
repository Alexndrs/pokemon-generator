from backend.model.unet import UNet
from backend.model.diffusion import GaussianDiffusion
import torch
import os
import numpy as np
import glob
from PIL import Image
from backend.data.preprocessing import PokemonDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_gif_from_frames(frames_dir, output_path, fps=10):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        print("Aucune image trouvée dans le dossier")
        return

    frame_files.reverse()

    # Charge les images
    frames = [Image.open(frame).convert("RGBA") for frame in frame_files]

    # Durée entre les frames en ms (Pillow attend ms, pas fps)
    duration = int(1000 / fps)

    # Sauvegarde en GIF animé
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 0 = boucle infinie
    )

    print(f"GIF créé : {output_path}")


def cleanup_frames(frames_dir):
    """
    Supprime tous les fichiers PNG du dossier frames_dir
    """
    frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
    
    for frame_file in frame_files:
        try:
            os.remove(frame_file)
        except OSError as e:
            print(f"Erreur lors de la suppression de {frame_file}: {e}")
    
    print(f"Nettoyage terminé : {len(frame_files)} images supprimées")

def generate_and_visualize(model_ckpt, sample_vid_dir, cond=None):

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


    image_size = 128
    batch_size = 1

    model = UNet(empty_condition_vector=empty_condition_vector).to(device)
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(model=model, device=device,empty_condition_vector=empty_condition_vector, H=image_size, W=image_size)

    print("Generating sample and saving intermediate steps...")
    # Generate a single image and save intermediate steps
    generated_image = diffusion.sample(
        batch_size=batch_size,
        channels=3,
        cond=cond,
        save_intermediate_steps=True,
        save_dir=sample_vid_dir,
        save_interval=9 
    )
    print("\nSample generation complete. Frames saved.")


    # Now, you'll use FFmpeg to create a video from these frames.
    # FFmpeg command will be run from the terminal after this script finishes.
    video_path = os.path.join(sample_vid_dir, "diffusion_process.gif")
    create_gif_from_frames(sample_vid_dir, video_path, fps=17)
    
    # Supprime les images PNG après création de la vidéo
    cleanup_frames(sample_vid_dir)

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "last_checkpoint.pt")
    sample_vid_dir = os.path.join(ROOT_DIR, "video_samples")
    os.makedirs(sample_vid_dir, exist_ok=True)
    print("ckpt_path :", ckpt_path, "sample_vid_dir :", sample_vid_dir)

    dataset = PokemonDataset(
        csv_path=os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv"),
        image_size=128,
        is_sprite=None,
        use_metadata=False,
        use_descriptions=False,
    )
    cond = dataset.encode_user_request(color="red", is_sprite=False).to(device)

    generate_and_visualize(ckpt_path,sample_vid_dir, cond=cond)