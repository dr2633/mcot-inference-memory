import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory.retrieval_faiss import FAISSRetriever
from reward_functions import reward_correctness, reward_efficiency, reward_formatting

# ----------------------------------------------
# Configuration
# ----------------------------------------------
MODEL_PATH = "grpo_checkpoints/final_model"
DATASET_NAME = "openai/gsm8k"
DATASET_SPLIT = "test"
OUTPUT_LOG_FILE = "rl_evaluation_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REWARD_WEIGHTS = {
    "correctness": 1.0,
    "efficiency": 0.5,
    "formatting": 0.3,
}

TOP_K_RETRIEVAL = 3  # Number of past memories to retrieve

# ----------------------------------------------
# Load Model & Tokenizer
# ----------------------------------------------
print(f"Loading GRPO-trained model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# ----------------------------------------------
# Load Dataset
# ----------------------------------------------
print(f"Loading dataset {DATASET_NAME} ({DATASET_SPLIT})...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.select(range(50))  # Limit for evaluation

# ----------------------------------------------
# Initialize Memory Retrieval
# ----------------------------------------------
retriever = FAISSRetriever(index_path="faiss_index")
retriever.load_index()

# ----------------------------------------------
# Evaluation Function
# ----------------------------------------------
def evaluate_model(model, dataset, use_memory=False):
    """
    Evaluates the model using standard or memory-augmented reasoning.

    Args:
        model (AutoModelForCausalLM): The trained model.
        dataset (Dataset): The dataset to evaluate.
        use_memory (bool): If True, retrieves past memory before answering.

    Returns:
        dict: Evaluation results.
    """
    results = []
    total_correct = 0

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        gold_answer = sample["answer"]

        retrieved_memories = retriever.retrieve_memory(question, top_k=TOP_K_RETRIEVAL) if use_memory else []
        retrieved_text = "\n".join([m["text"] for m in retrieved_memories]) if retrieved_memories else None

        # Construct prompt
        prompt = f"Question: {question}\nLet's think step by step."
        if use_memory and retrieved_text:
            prompt = f"Previous Reasoning:\n{retrieved_text}\n\n{prompt}"

        # Generate model response
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        generated_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if "Answer:" in generated_answer:
            predicted_answer = generated_answer.split("Answer:")[-1].strip()
        else:
            predicted_answer = generated_answer

        # Compute rewards
        correctness_score = reward_correctness([predicted_answer], [gold_answer])[0]
        efficiency_score = reward_efficiency([predicted_answer], [gold_answer])[0]
        formatting_score = reward_formatting([predicted_answer], retrieved_memories)[0]

        final_reward = (
            REWARD_WEIGHTS["correctness"] * correctness_score +
            REWARD_WEIGHTS["efficiency"] * efficiency_score +
            REWARD_WEIGHTS["formatting"] * formatting_score
        )
        final_reward = np.clip(final_reward, -1.0, 1.0)

        if correctness_score == 1.0:
            total_correct += 1

        results.append({
            "index": idx,
            "question": question,
            "gold_answer": gold_answer,
            "retrieved_memory": retrieved_text if use_memory else None,
            "generated_answer": generated_answer,
            "predicted_answer": predicted_answer,
            "correctness_score": correctness_score,
            "efficiency_score": efficiency_score,
            "formatting_score": formatting_score,
            "final_reward": final_reward,
        })

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx+1}/{len(dataset)} samples...")

    accuracy = total_correct / len(dataset)
    avg_correctness = np.mean([r["correctness_score"] for r in results])
    avg_efficiency = np.mean([r["efficiency_score"] for r in results])
    avg_formatting = np.mean([r["formatting_score"] for r in results])

    print("\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg Correctness Score: {avg_correctness:.3f}")
    print(f"Avg Efficiency Score: {avg_efficiency:.3f}")
    print(f"Avg Formatting Score: {avg_formatting:.3f}")

    return {
        "accuracy": accuracy,
        "avg_correctness": avg_correctness,
        "avg_efficiency": avg_efficiency,
        "avg_formatting": avg_formatting,
        "results": results,
    }

# ----------------------------------------------
# Run Evaluation
# ----------------------------------------------
print("\nRunning RL Model Evaluation...")
rl_results = evaluate_model(model, dataset, use_memory=True)

# ----------------------------------------------
# Save Results
# ----------------------------------------------
with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(rl_results, f, indent=4)

print(f"Evaluation results saved to {OUTPUT_LOG_FILE}")
