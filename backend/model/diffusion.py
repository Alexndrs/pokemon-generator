import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os
import math




class GaussianDiffusion:
    def __init__(self, model, device, timesteps=1000, beta_start=1e-4, beta_end=0.02, H=128, W=128):
        '''
        Implements the DDPM diffusion process scheduler and reverse sampling.

        Args:
            model (nn.Module): The noise prediction model (e.g., U-Net).
            device (torch.device): Device to run computations on.
            timesteps (int): Total number of diffusion steps T.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
        '''
        self.H = H
        self.W = W

        self.model = model
        self.device = device
        self.timesteps = timesteps

        # Linear schedule
        # self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)  # (T,)

        # Cosine schedule
        self.betas = self._get_cosine_schedule(timesteps).to(device)

        self.alphas = 1. - self.betas                             # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # (T,)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])  # (T,)

        # Useful precomputed constants for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)             # (T,)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)  # (T,)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)                 # (T,)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )  # (T,)

    def _get_cosine_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        s is a small offset to prevent alpha_bar from being too close to 1 at t=0
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Normalize to 1 at t=0
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999) 

    def _denormalize(self, img_tensor):
        """
        Denormalizes an image tensor from [-1, 1] to [0, 1] and converts to PIL Image.
        Expected input: (C, H, W) or (B, C, H, W)
        """
        # Ensure it's a clone and on CPU before operations
        img_tensor = img_tensor.clone().detach().cpu()

        # If it's a batch, pick the first image for visualization
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0] # Pick the first image in the batch

        # Denormalize from [-1, 1] to [0, 1]
        img_tensor = (img_tensor + 1) / 2
        img_tensor = torch.clamp(img_tensor, 0, 1) # Ensure values are strictly between 0 and 1

        # Convert to PIL Image
        to_pil = T.ToPILImage()
        return to_pil(img_tensor)


    def q_sample(self, x_start, t, noise=None):
        '''
        Forward diffusion (q): sample x_t given x_0.

        Args:
            x_start: clean image (B, C, H, W)
            t: timestep (B,)
            noise: optional noise (B, C, H, W)

        Returns:
            x_t: noised image (B, C, H, W)
            noise: the actual noise used (B, C, H, W)
        '''
        if noise is None:
            noise = torch.randn_like(x_start)

        # Add noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_ab * x_start + sqrt_1m_ab * noise
        return x_t, noise


    def predict_x0_from_eps(self, x_t, t, eps):
        '''
        Reconstruct x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)

        Args:
            x_t (Tensor): Noisy image at time t (B, C, H, W)
            t (Tensor): Time steps (B,)
            eps (Tensor): Predicted noise (B, C, H, W)

        Returns:
            x_0 (Tensor): Predicted denoised image (B, C, H, W)
        '''
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return (x_t - sqrt_one_minus_ab * eps) / sqrt_ab

    def p_sample(self, x_t, t):
        '''
        Perform one reverse sampling step: p(x_{t-1} | x_t)

        Args:
            x_t (Tensor): Image at step t (B, C, H, W)
            t (Tensor): Time steps (B,) or (B, 1)

        Returns:
            x_{t-1} (Tensor): Image at step t-1 (B, C, H, W)
        '''
        # Ensure t shape is (B,)
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Predict noise using the model
        eps = self.model(x_t, t.unsqueeze(-1))  # output: (B, C, H, W)
        # eps = self.model(x_t, t)  # output: (B, C, H, W)

        pred_x0 = self.predict_x0_from_eps(x_t, t, eps)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        # get constants of q(x_{t-1} | x_t, x_0)
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)

        model_mean = ( (torch.sqrt(alpha_cumprod_prev) * beta_t) / (1. - alpha_cumprod_t) ) * pred_x0 + \
                 ( (torch.sqrt(alpha_t) * (1. - alpha_cumprod_prev)) / (1. - alpha_cumprod_t) ) * x_t

        noise = torch.randn_like(x_t)
        variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return model_mean + nonzero_mask * torch.sqrt(variance) * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, save_intermediate_steps=False, save_dir=None, save_interval=10):
        '''
        Full reverse loop (sampling): from x_T ~ N(0,I) to x_0.
        Can optionally save intermediate steps for visualization.

        Args:
            shape: (B, C, H, W)
            save_intermediate_steps (bool): Whether to save intermediate images.
            save_dir (str, optional): Directory to save images if save_intermediate_steps is True.
                                      Defaults to a 'sampling_frames' folder.
            save_interval (int): Save an image every 'save_interval' steps.

        Returns:
            x_0: (B, C, H, W)

        '''
        img = torch.randn(shape, device=self.device)

        if save_intermediate_steps:
            if save_dir is None:
                save_dir = "sampling_frames"
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving intermediate sampling steps to: {save_dir}")

            # Save the initial noise image (x_T)
            initial_noise_img = self._denormalize(img)
            initial_noise_img.save(os.path.join(save_dir, f"step_{self.timesteps:04d}.png"))


        for t_idx in reversed(range(self.timesteps)): # Iterate through timesteps in reverse
            t_batch = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t_batch)

            # Save intermediate image
            if save_intermediate_steps and (t_idx % save_interval == 0 or t_idx == 0):
                # We need to detach and move to CPU for PIL
                display_img = self._denormalize(img)
                # Save with current timestep (t_idx) in filename
                display_img.save(os.path.join(save_dir, f"step_{t_idx:04d}.png"))
                print(f"\rSampling step {t_idx}/{self.timesteps} (Image saved)", end="", flush=True)
            elif t_idx % 100 == 0:
                print(f"\rSampling step {t_idx}/{self.timesteps}", end="", flush=True)

        print("\nSampling complete!")
        return img



    @torch.no_grad()
    def sample(self, batch_size, channels=3, save_intermediate_steps=False, save_dir=None, save_interval=10):
        '''
        Public method to generate samples.

        Returns:
            Tensor: (B, C, H, W)
        '''
        shape = (batch_size, channels, self.H, self.W)
        return self.p_sample_loop(shape, save_intermediate_steps, save_dir, save_interval)


    def loss(self, x_0, t):
        '''
        Compute training loss: L = MSE(eps_pred, eps)

        Args:
            x_0: clean image (B, C, H, W)
            t: timestep (B,)

        Returns:
            scalar loss
        '''
        x_t, noise = self.q_sample(x_0, t)
        eps_pred = self.model(x_t, t.unsqueeze(-1))  # (B, C, H, W)
        # eps_pred = self.model(x_t, t)  # (B, C, H, W)
        return nn.functional.mse_loss(eps_pred, noise)
