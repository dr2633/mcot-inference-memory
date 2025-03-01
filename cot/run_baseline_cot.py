#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_baseline_cot_multi_temp.py

Evaluates Qwen on the GSM8K dataset across different temperature settings.
Collects token usage, accuracy, and generation time metrics.
Generates violin and scatter plots for benchmarking output distributions.
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Chain-of-Thought on GSM8K with Qwen across temperatures")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Hugging Face model identifier.")
    parser.add_argument("--subset_size", type=int, default=50, help="Number of examples from GSM8K to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu).")
    parser.add_argument("--prompt_prefix", type=str, default="Let's reason through this step by step.",
                        help="Prompt prefix.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "data/qwen"),
                        help="Output directory.")

    return parser.parse_args()


# ----------------------------------------------
# Chain-of-Thought Prompt Construction
# ----------------------------------------------
def build_cot_prompt(question: str, prefix: str) -> str:
    return f"Question: {question}\n{prefix}\nShow all steps of your reasoning and conclude with '#### <integer>'.\n"


# ----------------------------------------------
# Inference & Logging Logic
# ----------------------------------------------
def generate_cot_and_answer(model, tokenizer, prompt, max_new_tokens, temperature, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_tokens = output_ids.shape[1] - input_tokens
    generation_time = end_time - start_time

    return output_text, input_tokens, output_tokens, generation_time


def extract_final_answer(generated_text: str) -> str:
    match = re.search(r'####\s*(\d+)', generated_text)
    return match.group(1) if match else "N/A"


def get_ground_truth_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(\d+)', answer_text)
    return match.group(1) if match else "N/A"


# ----------------------------------------------
# Main Experiment Loop
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
        print(f"\nRunning evaluation at temperature {temp} ...")
        results = []
        output_tokens_list = []
        total_output_tokens = 0
        total_generation_time = 0

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(args.output_dir, f"results_temp_{temp}_{timestamp}.json")

        for idx, sample in enumerate(ds):
            question = sample["question"]
            gold_solution = get_ground_truth_answer(sample["answer"].strip().lower())
            prompt = build_cot_prompt(question, args.prompt_prefix)

            generated_text, input_tokens, output_tokens, generation_time = generate_cot_and_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                device=args.device
            )
            predicted_answer = extract_final_answer(generated_text).strip().lower()
            normalized_predicted = re.sub(r"[^\d]", "", predicted_answer.strip())
            normalized_gold = re.sub(r"[^\d]", "", gold_solution.strip())
            is_correct = (normalized_predicted == normalized_gold)

            output_tokens_list.append(output_tokens)
            total_output_tokens += output_tokens
            total_generation_time += generation_time

            result_entry = {
                "index": idx,
                "temperature": temp,
                "question": question,
                "gold_solution": gold_solution,
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "output_tokens": output_tokens,
                "generation_time": generation_time
            }
            results.append(result_entry)

        avg_output_tokens = total_output_tokens / len(ds)
        token_stats[temp] = {"avg_output_tokens": avg_output_tokens, "output_tokens_list": output_tokens_list}

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)

        print(f"Results saved to {json_filename}")

    # Prepare Data for Visualization
    plot_data = []
    for temp, values in token_stats.items():
        for token_count in values["output_tokens_list"]:
            plot_data.append({"Temperature": temp, "Output Tokens": token_count})

    df = pd.DataFrame(plot_data)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Violin Plot - Output Token Distribution Across Temperatures
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Temperature", y="Output Tokens", data=df, palette="rocket")
    plt.title("Output Token Distribution Across Temperatures")
    plt.xlabel("Temperature")
    plt.ylabel("Output Tokens")
    violin_filename = os.path.join(args.output_dir, f"token_violin_plot_{timestamp}.png")
    plt.savefig(violin_filename)
    print(f"Violin plot saved to {violin_filename}")
    plt.show()

    # Scatter Plot - Average Output Tokens per Temperature
    plt.figure(figsize=(8, 5))
    scatter_data = pd.DataFrame({
        "Temperature": list(token_stats.keys()),
        "Avg Output Tokens": [values["avg_output_tokens"] for values in token_stats.values()]
    })
    sns.scatterplot(x="Temperature", y="Avg Output Tokens", data=scatter_data, palette="rocket", hue="Temperature",
                    size="Avg Output Tokens", legend=False)
    plt.title("Average Output Tokens per Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Average Output Tokens")
    scatter_filename = os.path.join(args.output_dir, f"token_scatter_plot_{timestamp}.png")
    plt.savefig(scatter_filename)
    print(f"Scatter plot saved to {scatter_filename}")
    plt.show()


if __name__ == "__main__":
    main()
