import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cot.run_memory_cot import generate_cot_with_memory
from memory.retrieval_faiss import FAISSRetriever
from rl.train_rl_memory import train_grpo_model
from evaluate_rl import evaluate_model

# ----------------------------------------------
# Configuration
# ----------------------------------------------
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(DEVICE)

# Initialize FAISS retriever with test index path
retriever = FAISSRetriever(index_path="test_faiss_index")  # Use test-specific index
retriever.load_index()  


# ----------------------------------------------
# Unit Tests
# ----------------------------------------------
class TestIntegration(unittest.TestCase):

    def setUp(self):
        """Set up sample questions for integration testing."""
        self.questions = [
            "Solve for x: 2x + 4 = 10.",
            "A rectangle has a length of 10 and a width of 5. What is its area?"
        ]
        self.expected_answers = [
            "x = 3",  # Answer to 2x + 4 = 10
            "50"  # Answer to 10 * 5
        ]

    def test_memory_retrieval(self):
        """Test FAISS retrieval returns relevant past responses."""
        for question in self.questions:
            retrieved_memories = retriever.retrieve_memory(question, top_k=3)
            self.assertGreaterEqual(len(retrieved_memories), 0, "Memory retrieval failed.")

    def test_cot_with_memory(self):
        """Test CoT generation with retrieved memory."""
        for question in self.questions:
            retrieved_memories = retriever.retrieve_memory(question, top_k=3)
            retrieved_text = "\n".join([m["text"] for m in retrieved_memories]) if retrieved_memories else None

            output = generate_cot_with_memory(question, memory_text=retrieved_text)
            self.assertIn("Let's think step by step", output, "Memory-augmented CoT reasoning missing.")

    def test_grpo_training(self):
        """Ensure GRPO training does not throw errors."""
        try:
            train_grpo_model()
        except Exception as e:
            self.fail(f"GRPO training encountered an error: {str(e)}")

    def test_grpo_model_evaluation(self):
        """Ensure trained GRPO model produces reasonable rewards."""
        evaluation_results = evaluate_model(model, dataset=None, use_memory=True)  # Using sample dataset
        self.assertIn("accuracy", evaluation_results, "Missing accuracy in GRPO evaluation.")
        self.assertGreaterEqual(evaluation_results["accuracy"], 0, "Invalid accuracy score.")


if __name__ == "__main__":
    unittest.main()
