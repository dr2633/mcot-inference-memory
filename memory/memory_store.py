# `memory_store.py`
#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
memory_store.py

Implements a simple FAISS-based memory store for text embeddings.
- Loads or creates a FAISS index.
- Stores embeddings + metadata for retrieval.
- Provides top-k similarity search to shape CoT reasoning.
"""

import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Union

# Example embedding approach (e.g., SentenceTransformers)
# If you prefer a Hugging Face model with Autotokenizer & AutoModel, adapt accordingly.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers if you're using the default embedding approach: pip install sentence-transformers")


class MemoryStore:
    def __init__(
        self,
        index_path: str = "faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        metadata_path: str = "metadata.pkl"
    ):
        """
        Args:
            index_path (str): Filepath or directory to store/retrieve the FAISS index files.
            embedding_model_name (str): Name of the embedding model to load.
            dimension (int): Embedding dimension used in the FAISS index.
            metadata_path (str): Path to store/retrieve metadata for each vector (user ID, text, etc.).
        """
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self.metadata_path = metadata_path

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

        # Initialize or load FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"[MemoryStore] Loaded existing FAISS index from {index_path}.")
        else:
            # Using a flat index for simplicity. For larger data, consider IVF or HNSW.
            self.index = faiss.IndexFlatL2(dimension)
            print("[MemoryStore] Created a new FAISS index (IndexFlatL2).")

        # Load metadata (stores user IDs, texts, etc. for each vector)
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[MemoryStore] Loaded existing metadata from {metadata_path}.")
        else:
            self.metadata = []
            print("[MemoryStore] Created a new metadata list.")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert input text to a normalized vector embedding.
        """
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        # Optionally L2 normalize for better approximate search
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        return embedding.astype(np.float32)

    def add_record(
        self,
        user_id: str,
        content: str,
        additional_info: Dict[str, Any] = None
    ) -> None:
        """
        Embeds the content, adds to FAISS index, and saves metadata.

        Args:
            user_id (str): Identifier for the user or session.
            content (str): The text chunk or user message to store.
            additional_info (dict): Extra metadata (e.g., timestamp, chain-of-thought steps).
        """
        vec = self.embed_text(content)
        # Reshape for Faiss (batch = 1)
        vec_2d = vec[np.newaxis, :]

        # Add vector to the index
        self.index.add(vec_2d)

        # Record metadata: store user_id, content, etc.
        record = {
            "user_id": user_id,
            "content": content,
        }
        if additional_info is not None:
            record.update(additional_info)

        self.metadata.append(record)
        print(f"[MemoryStore] Added record with user_id={user_id}. Number of records: {len(self.metadata)}.")

    def retrieve_top_k(
        self,
        query_text: str,
        user_id: str = None,
        k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Embed the query_text, perform a similarity search, return top-k metadata records.

        Args:
            query_text (str): The text to be embedded and searched against the index.
            user_id (str or None): If provided, you may filter results. (Example usage.)
            k (int): Number of records to return.

        Returns:
            A list of metadata dicts for the top-k nearest neighbors.
        """
        if len(self.metadata) == 0:
            print("[MemoryStore] Index is empty, returning no results.")
            return []

        query_vec = self.embed_text(query_text)
        query_vec_2d = query_vec[np.newaxis, :]
        distances, indices = self.index.search(query_vec_2d, k)

        # Flatten the index array
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                # Means no valid neighbor
                continue
            meta = self.metadata[idx]
            # (Optional) filter by user_id or other criteria
            if user_id and meta["user_id"] != user_id:
                # If you only want memory for the same user
                continue

            distance = distances[0][rank]
            result_entry = {
                "content": meta["content"],
                "user_id": meta["user_id"],
                "distance": float(distance)
            }
            # If additional metadata fields exist, merge them
            for k_, v_ in meta.items():
                if k_ not in result_entry:
                    result_entry[k_] = v_
            results.append(result_entry)

        return results

    def save_index_and_metadata(self) -> None:
        """
        Persist the current FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[MemoryStore] Index and metadata saved to {self.index_path} and {self.metadata_path}.")

    def rebuild_index(self) -> None:
        """
        Optional method to rebuild the FAISS index from existing metadata.
        - Might be needed if you change embedding or indexing strategies.
        """
        print("[MemoryStore] Rebuilding index from scratch...")
        new_index = faiss.IndexFlatL2(self.dimension)
        vectors = []
        for meta in self.metadata:
            vec = self.embed_text(meta["content"])
            vectors.append(vec)
        if len(vectors) > 0:
            vectors_np = np.vstack(vectors)
            new_index.add(vectors_np)
        self.index = new_index
        print("[MemoryStore] Rebuild complete.")

