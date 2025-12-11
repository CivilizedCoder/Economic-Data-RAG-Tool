#!/usr/bin/env python3

import os
import pickle
import numpy as np
import textwrap
import time
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
NEWS_DIR = "news_embeddings"
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
# ---------------------

def load_all_embeddings(directory):
    """
    Loads all .pkl files from the directory and aggregates them.
    Returns: (all_chunks, all_embeddings_matrix)
    """
    all_chunks = []
    all_embeddings = []

    print(f"Loading pre-computed embeddings from '{directory}'...")

    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return [], None

    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    if not files:
        print("No .pkl files found.")
        return [], None

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # data is a dict: {'chunks': [], 'embeddings': np.array, ...}
                all_chunks.extend(data['chunks'])
                all_embeddings.append(data['embeddings'])
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not all_embeddings:
        return [], None

    # Stack all individual embedding arrays into one large matrix
    combined_matrix = np.vstack(all_embeddings)

    print(f"Loaded {len(all_chunks)} chunks from {len(files)} files.")
    return all_chunks, combined_matrix

def search_index(query, model, all_chunks, all_embeddings, n=3):
    """Encodes query and compares against pre-loaded embeddings."""

    # BGE specific instruction
    instruction = "Represent this sentence for searching relevant passages: "
    query_with_instruction = f"{instruction}{query}"

    # Encode only the query
    query_embedding = model.encode(
        query_with_instruction,
        normalize_embeddings=True
    )

    # Vector comparison
    cosine_scores = util.cos_sim(query_embedding, all_embeddings)[0].cpu().numpy()

    # Get sorted indices (all of them, so we can filter)
    sorted_indices = np.argsort(cosine_scores)[::-1]

    results = []
    for idx in sorted_indices:
        chunk = all_chunks[idx]

        # --- Filter Logic ---
        # Skip chunks that look like just the metadata header
        clean_chunk = chunk.strip()
        if clean_chunk.startswith("Title:") or clean_chunk.endswith("+00:00"):
            continue
        if clean_chunk.startswith("Title:") and "Source:" in clean_chunk and "Date:" in clean_chunk and len(clean_chunk.split('\n')) <= 4:
            continue

        results.append({
            "chunk": chunk,
            "score": cosine_scores[idx].item()
        })

        # Stop once we have enough VALID results
        if len(results) >= n:
            break

    return results

def main():
    # 1. Load Model (Only needed for query encoding now)
    print("Loading model for query encoding...")
    try:
        model = SentenceTransformer(MODEL_NAME, device='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Data from Disk
    all_chunks, all_embeddings = load_all_embeddings(NEWS_DIR)

    if all_embeddings is None:
        print("No data available. Run the fetcher script first.")
        return

    # 3. Query Loop
    while True:
        print("-" * 30)
        query = input("Enter your question (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        start_time = time.perf_counter()

        results = search_index(query, model, all_chunks, all_embeddings, n=3)

        duration = time.perf_counter() - start_time

        print(f"\nFound relevant info in {duration:.4f} seconds:\n")

        if not results:
            print("No relevant chunks found (after filtering).")

        for i, res in enumerate(results):
            print(f"--- Result {i+1} (Score: {res['score']:.4f}) ---")
            print(textwrap.fill(res['chunk'], width=80))
            print()

if __name__ == "__main__":
    main()
