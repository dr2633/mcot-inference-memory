#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_baseline_cot.py

A baseline Chain-of-Thought (CoT) script for evaluating Qwen on the GSM8K dataset.
Collects token usage statistics, accuracy metrics, and generation time per sample.
"""

import argparse
import torch
import os
import json
import re
import time
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
        default=os.path.join(os.getcwd(), "data/qwen"),
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
    output_tokens = output_ids.shape[1] - input_tokens  # Only count newly generated tokens
    generation_time = end_time - start_time

    return output_text, input_tokens, output_tokens, generation_time

def extract_final_answer(generated_text: str) -> str:
    print(f"[DEBUG] Raw Model Output:\n{generated_text}\n", flush=True)

    match = re.search(r'####\s*(\d+)', generated_text)
    if match:
        return match.group(1)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
    if numbers:
        print(f"[DEBUG] Extracted Numbers: {numbers}", flush=True)
        return numbers[-1]

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

    total_tokens_used = 0
    total_generation_time = 0

    for idx, sample in enumerate(ds):
        question = sample["question"]
        gold_solution = get_ground_truth_answer(sample["answer"].strip().lower())
        prompt = build_cot_prompt(question, args.prompt_prefix)

        generated_text, input_tokens, output_tokens, generation_time = generate_cot_and_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )
        predicted_answer = extract_final_answer(generated_text).strip().lower()

        normalized_predicted = re.sub(r"[^\d]", "", predicted_answer.strip())
        normalized_gold = re.sub(r"[^\d]", "", gold_solution.strip())

        is_correct = (normalized_predicted == normalized_gold)

        print(f"\n[DEBUG] Sample {idx + 1}/{len(ds)}", flush=True)
        print(f"Question: {question}", flush=True)
        print(f"Prompt Used:\n{prompt}", flush=True)
        print(f"Generated Output:\n{generated_text}", flush=True)
        print(f"Extracted Answer: {predicted_answer}", flush=True)
        print(f"Expected Answer: {gold_solution}", flush=True)
        print(f"Correct?: {is_correct}", flush=True)
        print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Generation Time: {generation_time:.3f} sec", flush=True)
        print("-" * 80, flush=True)

        total_tokens_used += (input_tokens + output_tokens)
        total_generation_time += generation_time

        result_entry = {
            "index": idx,
            "question": question,
            "gold_solution": gold_solution,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "generation_time": generation_time
        }
        results.append(result_entry)

        with open(os.path.join(args.output_dir, f"sample_{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result_entry, f, indent=2)

    overall_accuracy = sum(r["is_correct"] for r in results) / len(results)
    avg_generation_time = total_generation_time / len(results)
    avg_tokens_per_sample = total_tokens_used / len(results)

    print("\nEvaluation Complete.")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Average Tokens Per Sample: {avg_tokens_per_sample:.2f}")
    print(f"Total Generation Time: {total_generation_time:.2f} sec")
    print(f"Average Generation Time Per Sample: {avg_generation_time:.3f} sec")

    summary = {
        "overall_accuracy": overall_accuracy,
        "total_tokens_used": total_tokens_used,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "total_generation_time": total_generation_time,
        "avg_generation_time_per_sample": avg_generation_time,
        "results": results
    }

    with open(os.path.join(args.output_dir, "baseline_cot_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output_dir}/baseline_cot_results.json")

if __name__ == "__main__":
    main()
