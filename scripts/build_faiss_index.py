"""
Build a FAISS index from extracted embeddings and metadata.

Examples:
python scripts/build_faiss_index.py \
  --metadata embeddings/resnet50/train/train_metadata.csv \
  --vector-type embedding \
  --index-out faiss_indices/resnet50_train.index

python scripts/build_faiss_index.py \
  --metadata embeddings/efficientnet_b0/train_concat/train_metadata.csv \
  --vector-type fusion \
  --index-out faiss_indices/efficientnet_b0_train_fusion.index
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Dict, List, Tuple
import faiss  # type: ignore
import numpy as np
import torch

def load_embeddings(metadata_csv: Path, vector_type: str = "embedding") -> Tuple[np.ndarray, List[Dict[str, str]]]:
    records, vectors = [], []
    with metadata_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            path_key = {"embedding": "embedding_path", "color": "color_path", "fusion": "fusion_path"}[vector_type]
            vec_path = Path(row.get(path_key, "") or "")
            if not vec_path.exists():
                continue
            if vec_path.suffix == ".pt":
                vec = torch.load(vec_path, map_location="cpu").detach().cpu().numpy()
            else:
                vec = np.load(vec_path)
            if vec.ndim > 1:
                vec = vec.reshape(-1)
            vec = vec.astype("float32")
            vec /= (np.linalg.norm(vec) + 1e-12)
            vectors.append(vec)
            records.append(row)
    if not vectors:
        raise RuntimeError(f"No vectors loaded for type '{vector_type}' from {metadata_csv}")
    return np.stack(vectors), records

def build_faiss_index(vectors: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu:
        if not faiss.get_num_gpus():
            raise RuntimeError("No GPU detected for FAISS GPU build")
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(vectors)
    return index

def save_artifacts(index: faiss.Index, meta: List[Dict[str, str]], index_out: Path, meta_out: Path) -> None:
    index_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss.index_gpu_to_cpu(index), str(index_out))
    with meta_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=meta[0].keys())
        writer.writeheader()
        writer.writerows(meta)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build FAISS index from embeddings.")
    p.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV from extraction.")
    p.add_argument("--vector-type", type=str, default="embedding", choices=["embedding", "color", "fusion"])
    p.add_argument("--index-out", type=Path, required=True, help="Where to save the FAISS index.")
    p.add_argument("--meta-out", type=Path, default=None, help="Optional rewritten metadata path.")
    p.add_argument("--gpu", action="store_true", help="Use GPU FAISS if available.")
    return p

def main() -> None:
    args = build_parser().parse_args()
    vectors, records = load_embeddings(args.metadata, vector_type=args.vector_type)
    index = build_faiss_index(vectors, use_gpu=args.gpu)
    meta_out = args.meta_out or args.index_out.with_suffix(".meta.csv")
    save_artifacts(index, records, args.index_out, meta_out)
    print(f"Built index with {index.ntotal} vectors of dim {vectors.shape[1]}")
    print(f"Index saved to: {args.index_out}")
    print(f"Metadata saved to: {meta_out}")

if __name__ == "__main__":
    main()
