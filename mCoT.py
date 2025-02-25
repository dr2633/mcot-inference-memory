import json
import os
from typing import List, Dict, Any

class MemoryStore:
    """
    Simple class to load user-specific memory from JSON and provide retrieval.
    Each JSON record is expected to have keys like: timestamp, user_feedback, preferences, etc.
    """
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.memory_data = []
        if os.path.exists(json_path):
            self.load_memory()
        else:
            print(f"No existing memory file found at {json_path}, starting fresh.")

    def load_memory(self):
        with open(self.json_path, 'r') as f:
            self.memory_data = json.load(f)

    def save_memory(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.memory_data, f, indent=2)

    def add_memory_entry(self, entry: Dict[str, Any]):
        self.memory_data.append(entry)
        self.save_memory()

    def retrieve_relevant_memory(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        A very naive retrieval: we simply look for records containing the query as a substring.
        In practice, you'll want to use a real retrieval approach (semantic search, etc.).
        """
        relevant = []
        for record in self.memory_data:
            if query.lower() in record.get("user_feedback", "").lower():
                relevant.append(record)
        # Sort or score them in more advanced ways if needed
        return relevant[:top_k]


class ChainOfThoughtReasoner:
    """
    Basic chain-of-thought reasoner that uses memory for personalization.
    """
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def run_cot(self, query: str, steps: int = 3) -> str:
        """
        A mock chain-of-thought approach that consults memory at each step
        and tries to adapt the response.
        """
        # 1. Retrieve relevant memory
        relevant_mem = self.memory_store.retrieve_relevant_memory(query)
        user_prefs = relevant_mem[0].get("preferences", {}) if relevant_mem else {}

        # 2. Initialize an internal chain-of-thought
        chain_steps = []

        for step in range(steps):
            # 3. "Reasoning" with memory (very simplified)
            # In practice, you'd use a model that conditions on both user prefs + the partial chain
            if user_prefs.get("formatting") == "latex_bold":
                chain_steps.append(f"(Step {step+1}) Your query was: \\textbf{{{query}}}")
            else:
                chain_steps.append(f"(Step {step+1}) Reasoning about: {query}")

        final_answer = f"Final answer for '{query}' with memory references."
        # 4. Optionally store new memory about this interaction
        new_memory_entry = {
            "timestamp": "2025-02-25",
            "user_feedback": "Used CoT for query",
            "preferences": user_prefs,
            "chain_steps": chain_steps
        }
        self.memory_store.add_memory_entry(new_memory_entry)

        # Combine chain steps in a single string for demonstration
        reasoning_trace = "\n".join(chain_steps)
        return reasoning_trace + "\n" + final_answer


if __name__ == "__main__":
    # Example usage
    memory_store = MemoryStore(json_path="user_memory.json")

    # Manually add a sample preference entry if none exist
    if not memory_store.memory_data:
        sample_entry = {
            "timestamp": "2025-02-24",
            "user_feedback": "LaTeX formatting please",
            "preferences": {
                "formatting": "latex_bold"
            }
        }
        memory_store.add_memory_entry(sample_entry)

    # Create reasoner
    cot_reasoner = ChainOfThoughtReasoner(memory_store)

    # Run CoT
    query = "How can you distinguish between different types of multivariate Hawkes processes?"
    result = cot_reasoner.run_cot(query, steps=2)
    print("Chain-of-Thought Output:\n", result)
