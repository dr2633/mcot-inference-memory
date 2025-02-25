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
import math
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
        default=0.7,
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
        "--output_file",
        type=str,
        default="baseline_cot_results.json",
        help="Path to store evaluation results in JSON format."
    )
    return parser.parse_args()

# ----------------------------------------------
# Chain-of-Thought Prompt Construction
# ----------------------------------------------
def build_cot_prompt(question: str, prefix: str) -> str:
    """
    Creates a simple CoT prompt by appending a reasoning prefix
    and instructing the model to produce step-by-step reasoning.
    """
    return f"Question: {question}\n{prefix}\n"

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
    """
    Generates a chain-of-thought response + final answer using the model.
    Splits the chain-of-thought from the final answer by searching for
    a delimiter (if desired), or just returns the entire generation.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def extract_final_answer(generated_text: str) -> str:
    """
    A naive approach to extract the final answer from the chain-of-thought.
    Here we assume the last line or a recognized pattern 'Answer:'.
    Modify as needed.
    """
    # If there's a marker like 'Answer:' or 'Final Answer:', parse it
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    # fallback to entire text if no distinct answer marker
    return generated_text.strip()

# ----------------------------------------------
# Main Script
# ----------------------------------------------
def main():
    args = parse_args()

    # 1. Load model & tokenizer
    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(args.device)

    # 2. Load GSM8K dataset
    print("Loading GSM8K dataset...")
    ds_full = load_dataset("openai/gsm8k", "main")
    ds = ds_full["train"]  # or ds_full["test"], depending on preference

    if not args.do_full_dataset:
        ds = ds.select(range(min(args.subset_size, len(ds))))

    print(f"Number of samples to evaluate: {len(ds)}")

    # 3. Evaluation placeholders
    results = []
    total_correct = 0

    # 4. Iterate over dataset
    for idx, sample in enumerate(ds):
        question = sample["question"]
        gold_solution = sample["answer"]  # The ground-truth final numeric/string answer

        # Build a chain-of-thought prompt
        prompt = build_cot_prompt(question, args.prompt_prefix)

        # Generate chain-of-thought + answer
        generated_text = generate_cot_and_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )

        # Extract final answer
        predicted_answer = extract_final_answer(generated_text)

        # Measure token usage
        # (1) Prompt tokens
        prompt_tokens = tokenizer(prompt)["input_ids"]
        # (2) Full output tokens
        output_tokens = tokenizer(generated_text)["input_ids"]
        # Optional: If you want to measure specifically the chain-of-thought portion,
        # you'd need a more advanced parsing strategy or a delimiter approach.

        # Check correctness
        # Basic numeric check if gold_solution is typically a string representing a number:
        # You may need a more robust compare function for complex answers
        is_correct = (predicted_answer.strip() == gold_solution.strip())

        if is_correct:
            total_correct += 1

        # Log the results
        result_entry = {
            "index": idx,
            "question": question,
            "gold_solution": gold_solution,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "prompt_length": len(prompt_tokens),
            "output_length": len(output_tokens),
        }
        results.append(result_entry)

        # Print live updates every few samples
        if (idx + 1) % 5 == 0:
            print(f"[{idx+1}/{len(ds)}] Current accuracy: {total_correct/(idx+1):.2%}")

    # 5. Compute final metrics
    overall_accuracy = total_correct / len(ds)
    avg_prompt_length = sum(r["prompt_length"] for r in results) / len(results)
    avg_output_length = sum(r["output_length"] for r in results) / len(results)

    print("\nEvaluation Complete.")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print(f"Average Prompt Token Length: {avg_prompt_length:.2f}")
    print(f"Average Output Token Length: {avg_output_length:.2f}")

    # 6. Save results to a JSON file
    import json
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "avg_prompt_length": avg_prompt_length,
            "avg_output_length": avg_output_length,
            "results": results
        }, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
