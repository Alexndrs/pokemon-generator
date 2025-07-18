import unittest
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from backend.model.diffusion import GaussianDiffusion
from backend.model.unet import UNet
from backend.data.preprocessing import PokemonDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestGaussianDiffusion(unittest.TestCase):
    def setUp(self):
        self.image_size = 128
        self.batch_size = 4
        self.timesteps = 1000

        # Mock model for testing
        self.model = UNet(in_channels=3, out_channels=3, time_emb_dim=256).to(device)
        self.diffusion = GaussianDiffusion(model=self.model, device=device, timesteps=self.timesteps, H=128, W=128)

        self.images = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(device)

    def test_q_sample(self):
        t = torch.randint(0, self.timesteps, (self.batch_size,), device=device)
        x_t, noise = self.diffusion.q_sample(self.images, t)
        self.assertEqual(x_t.shape, self.images.shape)
        self.assertEqual(noise.shape, self.images.shape)

    def test_predict_x0_from_eps(self):
        t = torch.randint(0, self.timesteps, (self.batch_size,), device=device)
        x_t, noise = self.diffusion.q_sample(self.images, t)
        eps_pred = self.model(x_t, t.unsqueeze(-1).float())

        x0_pred = self.diffusion.predict_x0_from_eps(x_t, t, eps_pred)
        self.assertEqual(x0_pred.shape, self.images.shape)

    def test_p_sample(self):
        t = torch.randint(1, self.timesteps, (self.batch_size,), device=device)
        x_t, _ = self.diffusion.q_sample(self.images, t)
        x_tm1 = self.diffusion.p_sample(x_t, t)
        self.assertEqual(x_tm1.shape, self.images.shape)

    def test_sample(self):
        samples = self.diffusion.sample(batch_size=2)
        self.assertEqual(samples.shape, (2, 3, 128, 128))
        self.assertFalse(torch.isnan(samples).any())

    def test_loss_computation(self):
        t = torch.randint(0, self.timesteps, (self.batch_size,), device=device)
        loss = self.diffusion.loss(self.images, t)
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss).any())


class TestDiffusionIntegrationWithRealDataset(unittest.TestCase):
    def test_end_to_end_on_real_data(self):
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv")

        dataset = PokemonDataset(csv_path=csv_path, image_size=128, use_metadata=False, use_descriptions=False)
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        images = batch["image"].to(device)

        model = UNet().to(device)
        diffusion = GaussianDiffusion(model=model, device=device, timesteps=1000)

        t = torch.randint(0, diffusion.timesteps, (images.size(0),), device=device)
        loss = diffusion.loss(images, t)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    print("\n==> Running diffusion tests...\n")
    unittest.main(verbosity=2)
