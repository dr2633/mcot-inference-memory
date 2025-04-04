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

    def setUp(self):
        self.questions = [
            "What is 12 multiplied by 8?",
            "If a train travels 60 miles per hour, how long will it take to travel 180 miles?"
        ]
        self.expected_answers = [
            "96",  
            "3 hours"  
        ]

    def test_cot_output(self):
        for question in self.questions:
            output = generate_cot_answer(question)
            self.assertIn("Let's think step by step", output, "CoT reasoning missing in output.")

    def test_cot_answer_extraction(self):
        for idx, question in enumerate(self.questions):
            output = generate_cot_answer(question)
            answer = output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()
            self.assertNotEqual(answer, "", "Generated answer is empty.")

    def test_token_length(self):
        for question in self.questions:
            output = generate_cot_answer(question, max_new_tokens=256)
            token_count = len(tokenizer(output)["input_ids"])
            self.assertLessEqual(token_count, 256, "Generated output exceeds token limit. Current limit is 256")


if __name__ == "__main__":
    unittest.main()
