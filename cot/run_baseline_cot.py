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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "data/qwen"),
                        help="Output directory.")
    parser.add_argument("--disable_flash_attn", action="store_true", help="Disable FlashAttention if supported.")
    parser.add_argument("--cot_prompt", type=str, default="Let's solve this step by step:",
                        help="Prompt to encourage chain-of-thought reasoning.")
    parser.add_argument("--temperatures", type=str, default="0,0.7",
                        help="Comma-separated list of temperature values to evaluate.")
    return parser.parse_args()


# ----------------------------------------------
# Coherence Evaluation Functions
# ----------------------------------------------
def calculate_perplexity(text, model, tokenizer, device="cuda"):
    """Computes perplexity (lower = more fluent/coherent)."""
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), inputs.input_ids[:, 1:].reshape(-1))
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return np.nan


def semantic_similarity(generated_text, gold_text):
    """Computes cosine similarity between generated and gold text embeddings."""
    try:
        gen_embedding = similarity_model.encode([generated_text])
        gold_embedding = similarity_model.encode([gold_text])
        return cosine_similarity(gen_embedding, gold_embedding)[0][0]
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return np.nan


def extract_answer(text):
    """Extract numerical answers from generated text."""
    # Try to find "the answer is X" or "= X" patterns
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s+(-?\d+(?:\.\d+)?)",
        r"(?:result|solution)\s+is\s+(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",
        r"=\s*(-?\d+(?:\.\d+)?)\s*\.",
        r"(-?\d+(?:\.\d+)?)\s*(?:is the answer|is our answer)"
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1]  # Take the last match as the final answer

    # Fallback: try to find the last number in the text
    all_numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if all_numbers:
        return all_numbers[-1]

    return None


def count_reasoning_steps(text):
    """Count the number of apparent reasoning steps in the text."""
    # Look for numbered steps, bullet points, or sentences that start with "First", "Then", etc.
    step_patterns = [
        r"^\s*\d+\.\s+",  # Numbered steps
        r"^\s*-\s+",  # Bullet points
        r"(?:first|second|third|fourth|fifth|next|then|finally)[,:]"  # Sequential markers
    ]

    steps = 0
    lines = text.lower().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern in step_patterns:
            if re.search(pattern, line):
                steps += 1
                break

    return max(1, steps)  # At least one step if any reasoning is present


def check_answer_correctness(extracted_answer, gold_solution):
    """Check if the extracted answer matches the gold solution."""
    if not extracted_answer:
        return False

    # Extract numbers from gold solution
    gold_numbers = re.findall(r"(-?\d+(?:\.\d+)?)", gold_solution)
    if not gold_numbers:
        return False

    # Convert to float for numerical comparison
    try:
        extracted_float = float(extracted_answer)
        gold_float = float(gold_numbers[-1])  # Assume the last number is the answer
        return abs(extracted_float - gold_float) < 1e-5  # Allow small floating-point differences
    except ValueError:
        # If conversion fails, fall back to string comparison
        return extracted_answer in gold_solution


# ----------------------------------------------
# Main Experiment
# ----------------------------------------------
def main():
    args = parse_args()

    print(f"Loading model {args.model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        )
        model.to(args.device)
        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading GSM8K dataset...")
    try:
        ds_full = load_dataset("openai/gsm8k", "main")
        ds = ds_full["train"].select(range(min(args.subset_size, len(ds_full["train"]))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse temperatures from command line argument
    temperatures = [float(t) for t in args.temperatures.split(',')]
    print(f"Evaluating at temperatures: {temperatures}")

    # Statistics for all temperatures
    all_temp_stats = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for temp in temperatures:
        print(f"\nEvaluating at temperature {temp} ...")
        results = []

        # Tracking metrics
        output_tokens_list = []
        perplexity_list = []
        similarity_list = []
        accuracy_list = []
        reasoning_steps_list = []

        json_filename = os.path.join(args.output_dir, f"results_temp_{temp}_{timestamp}.json")

        for idx, sample in enumerate(ds):
            try:
                question, gold_solution = sample["question"], sample["answer"].strip().lower()

                # Apply Chain-of-Thought prompt
                prompt = f"{question}\n{args.cot_prompt}"

                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

                with torch.no_grad():
                    # Use different generation settings based on temperature
                    if temp == 0:
                        # For temperature 0, use greedy decoding without sampling parameters
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,  # Use greedy decoding
                            pad_token_id=tokenizer.eos_token_id
                        )
                    else:
                        # For non-zero temperatures, enable sampling
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,  # Enable sampling
                            temperature=temp,
                            pad_token_id=tokenizer.eos_token_id
                        )

                generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                output_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]

                # Extract the answer and check correctness
                extracted_answer = extract_answer(generated_text)
                is_correct = check_answer_correctness(extracted_answer, gold_solution)

                # Count reasoning steps
                reasoning_steps = count_reasoning_steps(generated_text)

                # Compute evaluation metrics
                perplexity = calculate_perplexity(generated_text, model, tokenizer, args.device)
                similarity = semantic_similarity(generated_text, gold_solution)

                # Store metrics
                output_tokens_list.append(output_tokens)
                perplexity_list.append(perplexity)
                similarity_list.append(similarity)
                accuracy_list.append(is_correct)
                reasoning_steps_list.append(reasoning_steps)

                # Store detailed results
                results.append({
                    "index": idx,
                    "temperature": temp,
                    "question": question,
                    "gold_solution": gold_solution,
                    "generated_text": generated_text,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "reasoning_steps": reasoning_steps,
                    "output_tokens": output_tokens,
                    "perplexity": float(perplexity) if not np.isnan(perplexity) else None,
                    "semantic_similarity": float(similarity) if not np.isnan(similarity) else None
                })

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(ds)} examples")

            except Exception as e:
                print(f"Error processing example {idx} at temperature {temp}: {e}")
                continue

        # Calculate accuracy
        accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
        avg_reasoning_steps = sum(reasoning_steps_list) / len(reasoning_steps_list) if reasoning_steps_list else 0

        print(f"Temperature {temp} | Accuracy: {accuracy:.4f} | Avg Steps: {avg_reasoning_steps:.2f}")

        # Store stats for this temperature
        all_temp_stats[temp] = {
            "accuracy": accuracy,
            "avg_output_tokens": sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0,
            "avg_reasoning_steps": avg_reasoning_steps,
            "output_tokens_list": output_tokens_list,
            "perplexity_list": [float(p) if not np.isnan(p) else None for p in perplexity_list],
            "similarity_list": [float(s) if not np.isnan(s) else None for s in similarity_list],
            "reasoning_steps_list": reasoning_steps_list,
            "accuracy_list": accuracy_list
        }

        # Save detailed results
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)

        print(f"Results saved to {json_filename}")

    # Save summary statistics
    summary_file = os.path.join(args.output_dir, f"summary_stats_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_temp_stats.items()}, f, indent=2)  # Convert float keys to strings for JSON

    # Prepare Data for Visualization
    plot_data = []
    for temp, values in all_temp_stats.items():
        for i in range(len(values["output_tokens_list"])):
            try:
                # Skip NaN values
                perplexity = values["perplexity_list"][i]
                similarity = values["similarity_list"][i]
                if perplexity is None or similarity is None:
                    continue

                plot_data.append({
                    "Temperature": temp,
                    "Perplexity": perplexity,
                    "Semantic Similarity": similarity,
                    "Accuracy": int(values["accuracy_list"][i]),
                    "Reasoning Steps": values["reasoning_steps_list"][i]
                })
            except IndexError:
                continue  # Skip if any list is shorter

    if not plot_data:
        print("No data available for plotting")
        return

    df = pd.DataFrame(plot_data)
    sns.set(style="whitegrid")

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Violin Plot: Perplexity Across Temperatures
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Temperature", y="Perplexity", hue="Temperature", data=df, palette="rocket", legend=False)
    plt.title("Perplexity Across Temperatures (Lower = More Fluent)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"perplexity_violin_{timestamp}.png"))
    plt.close()

    # 2. Box Plot: Semantic Similarity vs Temperature
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Temperature", y="Semantic Similarity", data=df, palette="rocket")
    plt.title("Semantic Similarity Across Temperatures (Higher = More Coherent)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"similarity_box_{timestamp}.png"))
    plt.close()

    # 3. Line Plot: Accuracy Across Temperatures
    plt.figure(figsize=(10, 6))
    accuracy_data = {temp: stats["accuracy"] for temp, stats in all_temp_stats.items()}
    temps_sorted = sorted(accuracy_data.keys())
    plt.plot(temps_sorted, [accuracy_data[t] for t in temps_sorted], 'o-', linewidth=2)
    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"accuracy_line_{timestamp}.png"))
    plt.close()

    # 4. Bar Plot: Average Reasoning Steps
    plt.figure(figsize=(10, 6))
    steps_data = {temp: stats["avg_reasoning_steps"] for temp, stats in all_temp_stats.items()}
    temps_sorted = sorted(steps_data.keys())
    plt.bar(temps_sorted, [steps_data[t] for t in temps_sorted], color='skyblue')
    plt.xlabel("Temperature")
    plt.ylabel("Average Reasoning Steps")
    plt.title("Average Reasoning Steps vs Temperature")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"reasoning_steps_bar_{timestamp}.png"))
    plt.close()

    # Only create correlation heatmap if we have multiple temperatures
    if len(temperatures) > 1:
        # 5. Heatmap: Correlation between metrics
        plt.figure(figsize=(8, 6))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Between Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"correlation_heatmap_{timestamp}.png"))
        plt.close()

    print(f"All visualizations saved to {plots_dir}")

    # Print overall summary
    print("\n=== Overall Performance Summary ===")
    for temp in temperatures:
        print(f"Temperature {temp}:")
        print(f"  - Accuracy: {all_temp_stats[temp]['accuracy']:.4f}")
        print(f"  - Avg Tokens: {all_temp_stats[temp]['avg_output_tokens']:.1f}")
        print(f"  - Avg Steps: {all_temp_stats[temp]['avg_reasoning_steps']:.2f}")

    # Find best temperature for accuracy
    best_temp = max(all_temp_stats.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest temperature for accuracy: {best_temp} "
          f"({all_temp_stats[best_temp]['accuracy']:.4f})")


if __name__ == "__main__":
    main()