import unittest

import math
import torch
from torch import nn

from sinusoidal_embedding import SinusoidalEncoding


# the following code is copied verbatim from
# Hugging Face (https://huggingface.co/blog/annotated-diffusion)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MyTestCase(unittest.TestCase):
    def test_1d(self):
        # test my sinusoidal embedding is the same as HuggingFace's implementation
        # when the input tensor is 1d, i.e. of shape [N]

        batch_size = 100
        embedding_dim = 32

        # these layers are non-trainable
        my_layer = SinusoidalEncoding(embedding_dim, period_range=(math.tau, 10000 * math.tau))
        hf_layer = SinusoidalPositionEmbeddings(embedding_dim)

        with torch.inference_mode():
            for dtype in [torch.long, torch.int, torch.float, torch.double]:
                for device in [torch.device("cpu"), torch.device("cuda")]:
                    x = torch.arange(batch_size, dtype=dtype, device=device)
                    expected = hf_layer(x)
                    actual = my_layer(x)
                    # numerical error tolerance within 1e-5
                    self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))

    def test_hf_nd(self):
        # HuggingFace sinusoidal embedding doesn't generalize to higher dimensional inputs
        x = torch.rand(10, 5)
        hf_layer = SinusoidalPositionEmbeddings(32)
        with self.assertRaises(RuntimeError):
            hf_layer(x)


if __name__ == '__main__':
    unittest.main()
