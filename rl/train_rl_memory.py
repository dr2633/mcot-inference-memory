#!/usr/bin/env python
# coding: utf-8

"""
train_rl_memory.py

Implements GRPO-based reinforcement learning for memory-augmented Chain-of-Thought (mCoT).
Dynamically loads settings from grpo_config.yaml and integrates FAISS-based memory retrieval.
"""

import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from memory.retrieval_faiss import FAISSRetriever  # Import FAISS-based memory retrieval

# ----------------------------------------------
# Load GRPO Configuration
# ----------------------------------------------
def load_config(config_path="config/grpo_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# ----------------------------------------------
# Reward Function for GRPO
# ----------------------------------------------
def reward_function(completions, reference_answers, retrieved_memories=None):
    """
    Reward model for evaluating generated completions based on:
    - Correctness (if it matches the reference answer)
    - Efficiency (penalizing verbosity)
    - Formatting consistency (if it follows retrieved reasoning structures)
    """
    rewards = []
    for idx, completion in enumerate(completions):
        reward = 0.0
        gold_answer = reference_answers[idx].strip()

        # Correctness Reward
        if completion.strip() == gold_answer:
            reward += config["reward_weights"]["correctness"]

        # Efficiency Penalty: Reduces reward for excessive token usage
        efficiency_penalty = max(0, (len(completion.split()) - len(gold_answer.split())) / len(gold_answer.split()))
        reward -= efficiency_penalty * config["reward_weights"]["efficiency"]

        # Formatting Reward: Matches retrieved memory structure
        if retrieved_memories:
            memory_content = retrieved_memories[idx] if idx < len(retrieved_memories) else ""
            if memory_content and completion.startswith(memory_content[:50]):  # Alignment check
                reward += config["reward_weights"]["formatting"]

        # Clip rewards to prevent extreme values
        reward = max(config["clip_reward_range"][0], min(config["clip_reward_range"][1], reward))
        rewards.append(reward)

    return rewards

# ----------------------------------------------
# Main Training Function
# ----------------------------------------------
def main():
    # Set device
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Load model & tokenizer
    print(f"Loading model {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], trust_remote_code=True)
    model.to(device)

    # Load dataset
    print(f"Loading dataset {config['dataset_name']}...")
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    if config["eval_subset_size"] > 0:
        dataset = dataset.select(range(config["eval_subset_size"]))

    # Initialize FAISS-based memory retrieval (if enabled)
    memory_store = None
    if config["use_memory_retrieval"]:
        memory_store = FAISSRetriever(
            index_path=config["memory_config_path"],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        memory_store.load_index()

    # Initialize GRPO Trainer
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config["use_fp16"],
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start Training
    print("Starting GRPO training...")
    trainer.train()

    # Save model & push checkpoint
    trainer.save_model(config["output_dir"])
    print(f"Training complete. Model saved at {config['output_dir']}.")

if __name__ == "__main__":
    main()
