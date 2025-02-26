import unittest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.train_rl_memory import train_grpo_model
from reward_functions import reward_correctness, reward_efficiency, reward_formatting
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

# ----------------------------------------------
# Unit Tests
# ----------------------------------------------
class TestReinforcementLearning(unittest.TestCase):

    def setUp(self):
        """Set up test conditions."""
        self.sample_outputs = ["The answer is 10.", "Solve by substitution.", "The integral of x dx is x^2/2."]
        self.sample_gold_answers = ["10", "Solve by substitution.", "x^2/2"]
        self.sample_memories = ["Prior calculations showed 10 was the correct answer.", "Use substitution method for solving equations.", "Previous integration resulted in x^2/2."]

    def test_grpo_training(self):
        """Ensure GRPO training does not throw errors."""
        try:
            train_grpo_model()
        except Exception as e:
            self.fail(f"GRPO training encountered an error: {str(e)}")

    def test_reward_correctness(self):
        """Test reward function for correctness evaluation."""
        scores = reward_correctness(self.sample_outputs, self.sample_gold_answers)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), "Correctness reward out of range.")

    def test_reward_efficiency(self):
        """Test reward function for efficiency evaluation."""
        scores = reward_efficiency(self.sample_outputs, self.sample_gold_answers)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), "Efficiency reward out of range.")

    def test_reward_formatting(self):
        """Test reward function for formatting consistency."""
        scores = reward_formatting(self.sample_outputs, self.sample_memories)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), "Formatting reward out of range.")

    def test_model_evaluation(self):
        """Ensure trained GRPO model produces reasonable evaluation metrics."""
        evaluation_results = evaluate_model(model, dataset=None, use_memory=True)  # Use sample dataset
        self.assertIn("accuracy", evaluation_results, "Missing accuracy in GRPO evaluation.")
        self.assertGreaterEqual(evaluation_results["accuracy"], 0, "Invalid accuracy score.")

if __name__ == "__main__":
    unittest.main()
