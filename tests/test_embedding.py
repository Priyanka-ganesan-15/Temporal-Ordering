from pathlib import Path
import sys
import importlib

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OpenCLIPEmbedder = importlib.import_module("temporal_ordering.embedding").OpenCLIPEmbedder


class _DummyModel:
    def encode_image(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # Return a tiny deterministic embedding vector per image.
        return batch_tensor.mean(dim=(2, 3))


def _dummy_preprocess(image: Image.Image) -> torch.Tensor:
    value = image.getpixel((0, 0))[0] / 255.0
    return torch.full((3, 4, 4), value, dtype=torch.float32)


def _make_png(path: Path, grayscale_value: int) -> None:
    image = Image.new("RGB", (4, 4), color=(grayscale_value, grayscale_value, grayscale_value))
    image.save(path)


def test_embed_paths_writes_and_reads_cache(tmp_path: Path) -> None:
    img1 = tmp_path / "001.png"
    img2 = tmp_path / "002.png"
    _make_png(img1, 20)
    _make_png(img2, 200)

    embedder = OpenCLIPEmbedder.__new__(OpenCLIPEmbedder)
    embedder.device = "cpu"
    embedder.model = _DummyModel()
    embedder.preprocess = _dummy_preprocess

    cache_path = tmp_path / "emb.npy"
    first = embedder.embed_paths([img1, img2], batch_size=1, cache_path=cache_path, normalize=False)
    assert first.shape == (2, 3)
    assert cache_path.exists()

    # Change model to prove second call loads from cache rather than recomputing.
    embedder.model = None
    second = embedder.embed_paths([img1, img2], batch_size=1, cache_path=cache_path, normalize=False)
    assert np.allclose(first, second)
