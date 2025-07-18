import torch
import unittest
from backend.model.unet import SinusoidalTimeEmbedding, ConvBlock, UNet

# GPU config
print("=" * 50)
print("CONFIGURATION GPU")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU: {device}")

print("=" * 50)


class TestSinusoidalTimeEmbedding(unittest.TestCase):
    def test_output_shape(self):
        embedding_dim = 128
        B = 4
        t = torch.ones(B, 1, device=device)
        emb_layer = SinusoidalTimeEmbedding(embedding_dim).to(device)
        
        output = emb_layer(t)
        self.assertEqual(output.shape, (B, embedding_dim))
        self.assertEqual(output.device, device)

    def test_different_time_values(self):
        embedding_dim = 64
        B = 3
        t = torch.tensor([[1.0], [10.0], [100.0]], device=device)
        emb_layer = SinusoidalTimeEmbedding(embedding_dim).to(device)
        
        output = emb_layer(t)
        self.assertEqual(output.shape, (B, embedding_dim))
        self.assertEqual(output.device, device)
        # Verify that different time values produce different embeddings
        self.assertFalse(torch.equal(output[0], output[1]))


class TestConvBlock(unittest.TestCase):
    def test_forward_pass(self):
        B, C, H, W = 2, 3, 32, 32
        time_emb_dim = 64
        x = torch.randn(B, C, H, W, device=device)
        t_emb = torch.randn(B, time_emb_dim, device=device)

        block = ConvBlock(in_channels=3, out_channels=16, time_emb_dim=time_emb_dim).to(device)
        out = block(x, t_emb)
        
        self.assertEqual(out.shape, (B, 16, H, W))
        self.assertEqual(out.device, device)

    def test_same_input_output_channels(self):
        B, C, H, W = 2, 32, 64, 64
        time_emb_dim = 128
        x = torch.randn(B, C, H, W, device=device)
        t_emb = torch.randn(B, time_emb_dim, device=device)

        block = ConvBlock(in_channels=32, out_channels=32, time_emb_dim=time_emb_dim).to(device)
        out = block(x, t_emb)
        
        self.assertEqual(out.shape, (B, 32, H, W))
        self.assertEqual(out.device, device)


class TestUNet(unittest.TestCase):
    def test_forward_shape(self):
        B, C, H, W = 2, 3, 128, 128
        x = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.float32)

        model = UNet(in_channels=3, out_channels=3, time_emb_dim=256).to(device)
        out = model(x, t)
        
        self.assertEqual(out.shape, (B, 3, H, W))
        self.assertEqual(out.device, device)

    def test_different_output_channels(self):
        B, C, H, W = 1, 3, 128, 128
        x = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.float32)

        model = UNet(in_channels=3, out_channels=1, time_emb_dim=256).to(device)
        out = model(x, t)
        
        self.assertEqual(out.shape, (B, 1, H, W))
        self.assertEqual(out.device, device)

    def test_device_consistency(self):
        """Verify that all model parameters are on the correct device"""
        model = UNet().to(device)
        
        for name, param in model.named_parameters():
            self.assertEqual(param.device, device, f"Parameter {name} is not on {device}")

    def test_gradient_flow(self):
        """Test that gradients are computed correctly"""
        B, C, H, W = 1, 3, 128, 128
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)
        t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.float32)

        model = UNet().to(device)
        model.train()  # Mode training
        
        out = model(x, t)
        loss = out.sum() # Simple loss for testing purpose
        loss.backward()
        
        # Verify that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertEqual(x.grad.device, device)
        
        # Verify that all model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertEqual(param.grad.device, device, f"Gradient of {name} is not on {device}")

    def test_memory_usage(self):
        """Basic memory usage test to ensure model can run without OOM errors"""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(device)
            
            B, C, H, W = 4, 3, 128, 128
            x = torch.randn(B, C, H, W, device=device)
            t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.float32)
            
            model = UNet().to(device)
            with torch.no_grad():
                out = model(x, t)
            
            memory_after = torch.cuda.memory_allocated(device)
            memory_used = memory_after - memory_before
            
            print(f"Memory used: {memory_used / 1024**2:.2f} MB")
            self.assertGreater(memory_used, 0)

    def test_inference_mode(self):
        """Model should work in inference mode without gradients"""
        B, C, H, W = 1, 3, 128, 128
        x = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.float32)

        model = UNet().to(device)
        model.eval()
        
        with torch.no_grad():
            out = model(x, t)
        
        self.assertEqual(out.shape, (B, 3, H, W))
        self.assertEqual(out.device, device)

    def test_batch_processing(self):
        """Test model with different batch sizes"""
        model = UNet().to(device)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        C, H, W = 3, 128, 128
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, C, H, W, device=device)
                t = torch.randint(0, 1000, (batch_size, 1), device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    out = model(x, t)
                
                self.assertEqual(out.shape, (batch_size, 3, H, W))
                self.assertEqual(out.device, device)


# Integration test with dataset
try:
    from backend.data.preprocessing import PokemonDataset
    from torch.utils.data import DataLoader
    import os

    class TestWithDataset(unittest.TestCase):
        def test_integration_with_dataloader(self):
            ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            csv_path = os.path.join(ROOT_DIR, "data", "pokemon_dataset", "dataset.csv")

            dataset = PokemonDataset(csv_path=csv_path, image_size=128, use_metadata=False, use_descriptions=False)
            dataloader = DataLoader(dataset, batch_size=2)
            batch = next(iter(dataloader))

            x = batch["image"].to(device)
            t = torch.randint(0, 1000, (x.size(0), 1), device=device, dtype=torch.float32)

            model = UNet().to(device)
            with torch.no_grad():
                out = model(x, t)

            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.device, device)

except ImportError:
    print("Dataset test skipped (preprocessing not found).")


if __name__ == '__main__':
    print("\Starting tests...\n")
    unittest.main(verbosity=2)