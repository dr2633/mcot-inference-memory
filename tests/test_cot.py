import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from documentation.run_baseline_cot_1 import generate_cot_answer

# ----------------------------------------------
# Configuration
# ----------------------------------------------
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(DEVICE)


# ----------------------------------------------
# Unit Tests
# ----------------------------------------------
class TestChainOfThought(unittest.TestCase):
    """Test suite for Chain of Thought (CoT) reasoning functionality.
    Tests output format, answer extraction, and response length limits."""

    def setUp(self):
        """Initialize test data with arithmetic and word problems.
        Sets up questions and their expected numerical answers."""
        self.questions = [
            "What is 12 multiplied by 8?",
            "If a train travels 60 miles per hour, how long will it take to travel 180 miles?"
        ]
        self.expected_answers = [
            "96",  # Expected answer for 12 * 8
            "3 hours"  # Expected answer for 180 / 60
        ]

    def test_cot_output(self):
        """Verify that generated responses include step-by-step reasoning.
        Checks for 'Let's think step by step' phrase in output."""
        for question in self.questions:
            output = generate_cot_answer(question)
            self.assertIn("Let's think step by step", output, "CoT reasoning missing in output.")

    def test_cot_answer_extraction(self):
        """Verify that final answers can be extracted from responses.
        Extracts answer after 'Answer:' delimiter and checks for non-empty result."""
        for idx, question in enumerate(self.questions):
            output = generate_cot_answer(question)
            answer = output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()
            self.assertNotEqual(answer, "", "Generated answer is empty.")

    def test_token_length(self):
        """Verify response length stays within token limit.
        Ensures generated text doesn't exceed 256 tokens."""
        for question in self.questions:
            output = generate_cot_answer(question, max_new_tokens=256)
            token_count = len(tokenizer(output)["input_ids"])
            self.assertLessEqual(token_count, 256, "Generated output exceeds token limit. Current limit is 256")


if __name__ == "__main__":
    unittest.main()
