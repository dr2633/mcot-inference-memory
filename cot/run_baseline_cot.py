#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_baseline_cot.py

A baseline Chain-of-Thought (CoT) script for evaluating Qwen on the GSM8K dataset.
Collects token usage statistics and basic accuracy metrics.
"""

import argparse
import torch
import os
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Chain-of-Thought on GSM8K with Qwen")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen-7B",
        help="Hugging Face model identifier for Qwen or local path to a Qwen model."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=50,
        help="Number of examples from GSM8K to evaluate."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate for chain-of-thought and final answer."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)."
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="Let's reason through this step by step.",
        help="String used to initiate chain-of-thought reasoning."
    )
    parser.add_argument(
        "--do_full_dataset",
        action="store_true",
        help="Evaluate on the entire GSM8K dataset instead of subset_size."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "data/qwen"),  # Save relative to the script
        help="Directory to store evaluation JSON outputs."
    )

    return parser.parse_args()


# ----------------------------------------------
# Chain-of-Thought Prompt Construction
# ----------------------------------------------
def build_cot_prompt(question: str, prefix: str) -> str:
    return (
        f"Question: {question}\n"
        f"{prefix}\n"
        "Show all steps of your reasoning and conclude with the final answer in the format '#### <integer>'.\n"
    )

# ----------------------------------------------
# Inference & Logging Logic
# ----------------------------------------------
def generate_cot_and_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def extract_final_answer(generated_text: str) -> str:
    print(f"[DEBUG] Raw Model Output:\n{generated_text}\n", flush=True)

    # First, check for explicit '#### <integer>' format
    match = re.search(r'####\s*(\d+)', generated_text)
    if match:
        return match.group(1)

    # If not found, look for the last number in the output
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
    if numbers:
        print(f"[DEBUG] Extracted Numbers: {numbers}", flush=True)
        return numbers[-1]  # Take the last number as the final answer

    return "N/A"

def get_ground_truth_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(\d+)', answer_text)
    return match.group(1) if match else "N/A"

# ----------------------------------------------
# Main Script
# ----------------------------------------------
def main():
    args = parse_args()

    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(args.device)

    print("Loading GSM8K dataset...")
    ds_full = load_dataset("openai/gsm8k", "main")
    ds = ds_full["train"]

    if not args.do_full_dataset:
        ds = ds.select(range(min(args.subset_size, len(ds))))

    print(f"Number of samples to evaluate: {len(ds)}")
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, sample in enumerate(ds):
        question = sample["question"]
        gold_solution = get_ground_truth_answer(sample["answer"].strip().lower())
        prompt = build_cot_prompt(question, args.prompt_prefix)
        generated_text = generate_cot_and_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )
        predicted_answer = extract_final_answer(generated_text).strip().lower()
        # Normalize answers by stripping spaces, newlines, and punctuation
        normalized_predicted = re.sub(r"[^\d]", "", predicted_answer.strip())  # Keep only numbers
        normalized_gold = re.sub(r"[^\d]", "", gold_solution.strip())  # Keep only numbers

        is_correct = (normalized_predicted == normalized_gold)

        # Print Model Output for Debugging
        print(f"\n[DEBUG] Sample {idx + 1}/{len(ds)}", flush=True)
        print(f"Question: {question}", flush=True)
        print(f"Prompt Used:\n{prompt}", flush=True)
        print(f"Generated Output:\n{generated_text}", flush=True)
        print(f"Extracted Answer: {predicted_answer}", flush=True)
        print(f"Expected Answer: {gold_solution}", flush=True)
        print(f"Correct?: {is_correct}", flush=True)
        print("-" * 80, flush=True)

        # Save per-sample JSON
        os.makedirs(args.output_dir, exist_ok=True)
        result_entry = {
            "index": idx,
            "question": question,
            "gold_solution": gold_solution,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        }
        results.append(result_entry)

        with open(os.path.join(args.output_dir, f"sample_{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result_entry, f, indent=2)

    overall_accuracy = sum(r["is_correct"] for r in results) / len(results)
    print("\nEvaluation Complete.")
    print(f"Accuracy: {overall_accuracy:.2%}")

    with open(os.path.join(args.output_dir, "baseline_cot_results.json"), "w", encoding="utf-8") as f:
        json.dump({"overall_accuracy": overall_accuracy, "results": results}, f, indent=2)
    print(f"Results saved to {args.output_dir}/baseline_cot_results.json")

if __name__ == "__main__":
    main()
