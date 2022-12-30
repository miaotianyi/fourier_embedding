# fourier-embedding
Generalized Fourier / sinusoidal embedding modules for arbitrary real-valued tensor inputs in PyTorch

Each of the following files contains a standalone embedding layer:
- `sinusoidal_embedding.py`:
The `SinusoidalEmbedding` layer has the same input-output interface `(*)->(*, H)`
as PyTorch's `nn.Embedding` layer, except it's not trainable.
It implements a generalized version of positional encoding in [Transformer](https://arxiv.org/abs/1706.03762),
supports more wavelength options, and accepts input tensors of arbitrary shape.
