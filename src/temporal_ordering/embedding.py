"""OpenCLIP embedding APIs and CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from temporal_ordering.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_PRETRAINED,
    resolve_project_path,
)
from temporal_ordering.data_loader import load_sequences
from temporal_ordering.utils import flatten_frames, get_sequence_or_raise


class OpenCLIPEmbedder:
    """Compute OpenCLIP image embeddings for a list of image paths."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip is required. Install with: pip install open_clip_torch"
            ) from exc
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess

    def embed_paths(
        self,
        image_paths: list[str | Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_path: str | Path | None = None,
        normalize: bool = True,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Embed images and return array of shape (n_images, embedding_dim)."""
        if not image_paths:
            raise ValueError("image_paths must contain at least one path")

        if cache_path is not None:
            cache = Path(cache_path)
            if cache.exists() and not force_recompute:
                return np.load(cache)
        else:
            cache = None

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                batch_images = []
                for path in batch_paths:
                    with Image.open(path) as img:
                        batch_images.append(self.preprocess(img.convert("RGB")))

                batch_tensor = torch.stack(batch_images).to(self.device)
                feats = self.model.encode_image(batch_tensor)
                if normalize:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                outputs.append(feats.cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(outputs, axis=0)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings)
        return embeddings

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed text prompts and return array of shape (n_texts, embedding_dim)."""
        if not texts:
            raise ValueError("texts must contain at least one prompt")

        import open_clip

        with torch.no_grad():
            tokens = open_clip.tokenize(texts).to(self.device)
            feats = self.model.encode_text(tokens)
            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)


def parse_embedder_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OpenCLIP embeddings for frame images")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to sequences manifest")
    parser.add_argument("--sequence", default=None, help="Embed only one sequence_id")
    parser.add_argument("--all", action="store_true", help="Embed all frames across all sequences")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Directory for cached .npy outputs")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write cache files")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if cache exists")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenCLIP model name")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag")
    return parser.parse_args()


def run_embedder_cli(args: argparse.Namespace) -> None:
    if not args.all and not args.sequence:
        raise ValueError("Specify either --sequence <id> or --all")

    sequences = load_sequences(args.manifest)
    embedder = OpenCLIPEmbedder(model_name=args.model, pretrained=args.pretrained)
    cache_dir = resolve_project_path(args.cache_dir)

    if args.sequence:
        seq = get_sequence_or_raise(sequences, args.sequence)
        cache_path = None if args.no_cache else cache_dir / f"{seq.sequence_id}.npy"
        embeddings = embedder.embed_paths(
            seq.frames,
            batch_size=args.batch_size,
            cache_path=cache_path,
            force_recompute=args.force,
        )
        print(f"{seq.sequence_id}: embedded {len(seq.frames)} frames -> {embeddings.shape}")

    if args.all:
        all_frames = flatten_frames(sequences)
        cache_path = None if args.no_cache else cache_dir / "all_frames.npy"
        embeddings = embedder.embed_paths(
            all_frames,
            batch_size=args.batch_size,
            cache_path=cache_path,
            force_recompute=args.force,
        )
        print(f"all_sequences: embedded {len(all_frames)} frames -> {embeddings.shape}")


def main() -> None:
    run_embedder_cli(parse_embedder_args())


if __name__ == "__main__":
    main()
