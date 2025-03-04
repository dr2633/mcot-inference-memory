#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_baseline_cot_coherence.py

Evaluate on GSM8K dataset across temperature settings.
Collects accuracy, token usage, coherence (perplexity, semantic similarity, entity overlap).
Generates violin and scatter plots for coherence benchmarking.
"""

import argparse
import torch
import os
import json
import re
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Load models for evaluation

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Coherence Evaluation of Chain-of-Thought on GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Hugging Face model identifier.")
    parser.add_argument("--subset_size", type=int, default=50, help="Number of examples from GSM8K to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "data/qwen"), help="Output directory.")
    parser.add_argument("--disable_flash_attn", action="store_true", help="Disable FlashAttention if supported.")
    return parser.parse_args()

# ----------------------------------------------
# Coherence Evaluation Functions
# ----------------------------------------------
def calculate_perplexity(text, model, tokenizer, device="cuda"):
    """Computes perplexity (lower = more fluent/coherent)."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), inputs.input_ids[:, 1:].reshape(-1))
    return torch.exp(loss).item()

def semantic_similarity(generated_text, gold_text):
    """Computes cosine similarity between generated and gold text embeddings."""
    gen_embedding = similarity_model.encode([generated_text])
    gold_embedding = similarity_model.encode([gold_text])
    return cosine_similarity(gen_embedding, gold_embedding)[0][0]

# ----------------------------------------------
# Main Experiment
# ----------------------------------------------
def main():
    args = parse_args()

    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(args.device)

    print("Loading GSM8K dataset...")
    ds_full = load_dataset("openai/gsm8k", "main")
    ds = ds_full["train"].select(range(min(args.subset_size, len(ds_full["train"]))))

    os.makedirs(args.output_dir, exist_ok=True)
    temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1]
    token_stats = {}

    for temp in temperatures:
        print(f"\nEvaluating at temperature {temp} ...")
        results = []
        output_tokens_list, perplexity_list, similarity_list, entity_overlap_list = [], [], [], []

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(args.output_dir, f"results_temp_{temp}_{timestamp}.json")

        for idx, sample in enumerate(ds):
            question, gold_solution = sample["question"], sample["answer"].strip().lower()

            inputs = tokenizer(question, return_tensors="pt").to(args.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=temp)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]

            # Compute evaluation metrics
            perplexity = calculate_perplexity(generated_text, model, tokenizer, args.device)
            similarity = semantic_similarity(generated_text, gold_solution)

            # Store results
            output_tokens_list.append(output_tokens)
            perplexity_list.append(perplexity)
            similarity_list.append(similarity)

            results.append({
                "index": idx, "temperature": temp, "question": question,
                "gold_solution": gold_solution, "generated_text": generated_text,
                "output_tokens": output_tokens, "perplexity": perplexity,
            })

        # Store stats
        token_stats[temp] = {
            "avg_output_tokens": sum(output_tokens_list) / len(output_tokens_list),
            "output_tokens_list": output_tokens_list,
            "perplexity_list": perplexity_list,
            "similarity_list": similarity_list,
            "entity_overlap_list": entity_overlap_list
        }

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)

        print(f"Results saved to {json_filename}")

    # Prepare Data for Visualization
    plot_data = []
    for temp, values in token_stats.items():
        for i in range(len(values["output_tokens_list"])):
            plot_data.append({"Temperature": temp, "Perplexity": values["perplexity_list"][i], "Semantic Similarity": values["similarity_list"][i]})

    df = pd.DataFrame(plot_data)
    sns.set(style="whitegrid")

    # Violin Plot: Perplexity Across Temperatures
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Temperature", y="Perplexity", data=df, palette="rocket")
    plt.title("Perplexity Across Temperatures (Lower = More Fluent)")
    plt.savefig(os.path.join(args.output_dir, f"perplexity_violin_{timestamp}.png"))
    plt.show()

    # Scatter Plot: Semantic Similarity Across Temperatures
    plt.figure(figsize=(8, 5))
    scatter_df = df.groupby("Temperature")["Semantic Similarity"].mean().reset_index()
    sns.scatterplot(x="Temperature", y="Semantic Similarity", data=scatter_df, palette="rocket", hue="Temperature", size="Semantic Similarity", legend=False)
    plt.title("Semantic Similarity vs. Temperature (Higher = More Coherent)")
    plt.savefig(os.path.join(args.output_dir, f"similarity_scatter_{timestamp}.png"))
    plt.show()

if __name__ == "__main__":
    main()
