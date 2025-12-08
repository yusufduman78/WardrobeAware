"""
Quick sanity check for extracted embeddings.

Usage:
python notebooks/embedding_sanity.py \
  --metadata embeddings/resnet50/train/train_metadata.csv \
  --max-samples 5000 \
  --num-queries 5 \
  --top-k 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tabulate import tabulate


def load_embeddings(metadata_path: Path, max_samples: int | None) -> Tuple[torch.Tensor, List[dict]]:
    df = pd.read_csv(metadata_path)
    if max_samples is not None and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    emb_list: List[torch.Tensor] = []
    records: List[dict] = []
    for _, row in df.iterrows():
        emb_path = Path(row["embedding_path"])
        if not emb_path.exists():
            continue
        emb = torch.load(emb_path, map_location="cpu")
        emb_list.append(emb)
        records.append(row.to_dict())

    if not emb_list:
        raise RuntimeError("No embeddings loaded. Check paths in the metadata file.")

    matrix = torch.stack(emb_list)
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    return matrix, records


def nearest_neighbors(
    matrix: torch.Tensor, records: List[dict], num_queries: int, top_k: int
) -> List[Tuple[dict, List[Tuple[dict, float]]]]:
    # pick random queries
    idxs = random.sample(range(matrix.size(0)), k=min(num_queries, matrix.size(0)))
    sims = matrix @ matrix.T

    results = []
    for q_idx in idxs:
        scores = sims[q_idx]
        values, indices = torch.topk(scores, k=min(top_k + 1, scores.numel()))
        neighbors = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            if idx == q_idx:
                continue  # skip self
            neighbors.append((records[idx], float(score)))
            if len(neighbors) >= top_k:
                break
        results.append((records[q_idx], neighbors))
    return results


def print_report(results: List[Tuple[dict, List[Tuple[dict, float]]]]) -> None:
    for query, neighbors in results:
        print("\nQuery:", query["image_path"])
        print(f"Category: {query['category_id']} ({query.get('category_name', '')})")
        table = []
        for nbr, score in neighbors:
            table.append(
                [
                    nbr["image_path"],
                    f"{nbr['category_id']} ({nbr.get('category_name', '')})",
                    f"{score:.4f}",
                ]
            )
        print(tabulate(table, headers=["neighbor_image", "category", "cosine"], tablefmt="github"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check nearest neighbors on extracted embeddings.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV from extraction.")
    parser.add_argument("--max-samples", type=int, default=5000, help="Cap number of embeddings to load for speed.")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of random queries to test.")
    parser.add_argument("--top-k", type=int, default=5, help="Neighbors to display.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    matrix, records = load_embeddings(args.metadata, args.max_samples)
    results = nearest_neighbors(matrix, records, args.num_queries, args.top_k)
    print_report(results)


if __name__ == "__main__":
    main()
