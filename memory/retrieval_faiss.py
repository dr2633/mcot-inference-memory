import faiss
import os
import pickle
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FAISSRetriever:
    """
    A FAISS-based retrieval system for memory-augmented Chain-of-Thought (mCoT).
    Stores reasoning steps as dense embeddings and retrieves top-K relevant memories.
    """

    def __init__(self, index_path="faiss_index", metadata_path="metadata.pkl",
                 embedding_model="sentence-transformers/all-MiniLM-L12-v2", dim=384, use_gpu=False):
        """
        Initializes the FAISS index, embedding model, and metadata storage.

        Args:
            index_path (str): Path to store FAISS index.
            metadata_path (str): Path to store memory metadata (raw text entries).
            embedding_model (str): Sentence Transformer model for text embeddings.
            dim (int): Dimension of embeddings (should match model output).
            use_gpu (bool): Whether to use FAISS-GPU (requires CUDA).
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.use_gpu = use_gpu
        self.dim = dim

        # Initialize the SentenceTransformer with device selection
        device = "cuda" if use_gpu else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

        # Load or create FAISS index using inner product (IP) for cosine similarity.
        if os.path.exists(self.index_path):
            logging.info(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
        else:
            logging.info("Creating new FAISS index using inner product (cosine similarity)...")
            self.index = faiss.IndexFlatIP(dim)  # Use inner product; works for normalized embeddings
            if self.use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Loads stored metadata (reasoning steps) if available."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Failed to load metadata: {e}")
                return {}
        return {}

    def add_memory(self, user_id: str, reasoning_text: str) -> None:
        """
        Adds a new memory entry for a specific user.

        Args:
            user_id (str): Unique user identifier.
            reasoning_text (str): Chain-of-thought reasoning step to store.
        """
        # Encode text to embedding with normalization
        embedding = self.embedding_model.encode([reasoning_text], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)

        # Assign index ID based on current metadata count
        entry_id = len(self.metadata)

        # Add embedding to FAISS index
        self.index.add(embedding)

        # Store metadata
        self.metadata[entry_id] = {"user_id": user_id, "text": reasoning_text}

        logging.info(f"Added memory {entry_id} for user {user_id}")

    def retrieve_memory(self, query_text: str, user_id: str = None, top_k: int = 3, min_similarity: float = 0.7):
        """
        Retrieves the most relevant past reasoning steps for a query.

        Args:
            query_text (str): The current problem/query.
            user_id (str): If provided, restrict retrieval to this user’s memory.
            top_k (int): Number of memories to retrieve.
            min_similarity (float): Minimum cosine similarity threshold.

        Returns:
            List[Dict]: Retrieved reasoning steps with similarity scores.
        """
        if self.index.ntotal == 0:
            return []  # No stored memories yet

        # Encode query to embedding
        query_embedding = self.embedding_model.encode([query_text], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Retrieve nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_memories = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # With an IP index and normalized embeddings, the distance returned is the cosine similarity.
            similarity = distances[0][i]
            if idx in self.metadata and similarity >= min_similarity:
                memory_entry = self.metadata[idx]
                # If user-specific retrieval is enabled, check user_id match
                if user_id and memory_entry["user_id"] != user_id:
                    continue
                retrieved_memories.append({
                    "user_id": memory_entry["user_id"],
                    "text": memory_entry["text"],
                    "similarity": round(similarity, 3)
                })
        return retrieved_memories

    def save(self) -> None:
        """Saves FAISS index and metadata for persistence."""
        # If using GPU, convert index back to CPU before saving.
        if self.use_gpu:
            self.index = faiss.index_gpu_to_cpu(self.index)
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logging.info(f"Saved FAISS index to {self.index_path} and metadata to {self.metadata_path}.")
        except Exception as e:
            logging.error(f"Error saving index or metadata: {e}")

    def reset_memory(self) -> None:
        """Clears all stored memories and resets the FAISS index."""
        self.index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.metadata = {}
        logging.info("Memory store reset. Previous data cleared.")


if __name__ == "__main__":
    # Example usage:
    retriever = FAISSRetriever(use_gpu=False)
    retriever.add_memory("user1", "This is a sample reasoning step for user1.")
    retriever.add_memory("user2", "Another reasoning step from user2.")

    # Retrieve memories for user1 that are relevant to the query.
    results = retriever.retrieve_memory("sample reasoning", user_id="user1")
    logging.info("Retrieved memories: %s", results)

    retriever.save()
