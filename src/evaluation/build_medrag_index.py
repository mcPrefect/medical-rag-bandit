"""Build FAISS index over MedRAG textbook corpus."""

import json
import numpy as np
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss


def main():
    output_dir = Path("data/medrag")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MedRAG textbooks corpus...")
    corpus = load_dataset("MedRAG/textbooks", split="train")
    print(f"Loaded {len(corpus)} chunks")

    texts = [chunk['contents'] for chunk in corpus]
    ids   = [chunk['id'] for chunk in corpus]

    print("Saving chunk texts...")
    with open(output_dir / "textbooks_chunks.json", "w") as f:
        json.dump({"ids": ids, "texts": texts}, f)
    print(f"Saved {len(texts)} chunks")

    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # for cosine similarity via inner product
    )
    print(f"Encoded {len(embeddings)} chunks, shape: {embeddings.shape}")

    np.save(output_dir / "textbooks_embeddings.npy", embeddings)
    print("Embeddings saved")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vectors
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(output_dir / "textbooks_index.faiss"))
    print(f"FAISS index saved, {index.ntotal} vectors")

    print("\nDone. Files saved to data/medrag/:")
    print("  textbooks_index.faiss")
    print("  textbooks_chunks.json")
    print("  textbooks_embeddings.npy")


if __name__ == "__main__":
    main()