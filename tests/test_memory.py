import unittest
import os
import faiss
import numpy as np
from memory.retrieval_faiss import FAISSRetriever

# ----------------------------------------------
# Configuration
# ----------------------------------------------
INDEX_PATH = "test_faiss_index"
TEST_QUERIES = [
    "What is 2 + 2?",
    "Solve for x: 3x + 6 = 15.",
    "A circle has a radius of 4. What is its area?"
]

TEST_MEMORIES = [
    {"text": "2 + 2 equals 4."},
    {"text": "Rearrange the equation: 3x = 9, so x = 3."},
    {"text": "Area of a circle: πr^2. With r=4, area = 16π."}
]

# ----------------------------------------------
# Unit Tests
# ----------------------------------------------
class TestFAISSRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize FAISS retriever and populate test index."""
        cls.retriever = FAISSRetriever(index_path=INDEX_PATH)
        cls.retriever.build_index([m["text"] for m in TEST_MEMORIES])

    def test_index_creation(self):
        """Verify that FAISS index is successfully created and saved."""
        self.assertTrue(os.path.exists(INDEX_PATH + ".index"), "FAISS index file not created.")

    def test_memory_retrieval(self):
        """Test retrieval of relevant memory."""
        for idx, query in enumerate(TEST_QUERIES):
            retrieved_memories = self.retriever.retrieve_memory(query, top_k=1)
            self.assertGreater(len(retrieved_memories), 0, "Memory retrieval failed.")
            self.assertIn(TEST_MEMORIES[idx]["text"], retrieved_memories[0]["text"], "Incorrect memory retrieved.")

    def test_memory_update(self):
        """Test updating FAISS index with new memory."""
        new_memory = "A square has 4 equal sides."
        self.retriever.add_to_index([new_memory])
        retrieved_memories = self.retriever.retrieve_memory("What are the properties of a square?", top_k=1)
        self.assertIn(new_memory, retrieved_memories[0]["text"], "Memory update failed.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test FAISS index."""
        if os.path.exists(INDEX_PATH + ".index"):
            os.remove(INDEX_PATH + ".index")

if __name__ == "__main__":
    unittest.main()
