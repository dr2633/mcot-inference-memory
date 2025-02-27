#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_rag_cot.py

An improved Chain-of-Thought (CoT) script for evaluating Qwen on the GSM8K dataset
with Retrieval-Augmented Generation (RAG). Incorporates memory conditioning via
retrieved student profiles to personalize explanations.
"""

import argparse
import torch
import math
import datetime
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate

# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RAG-Conditioned Chain-of-Thought on GSM8K with Qwen")
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
        "--output_file",
        type=str,
        default=f"rag_cot_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to store evaluation results in JSON format."
    )
    return parser.parse_args()

# ----------------------------------------------
# Chain-of-Thought Prompt Construction
# ----------------------------------------------
def build_cot_prompt(question: str, prefix: str, retrieved_profile: dict = None) -> str:
    """
    Constructs a chain-of-thought prompt with optional student profile memory.
    """
    if retrieved_profile:
        profile_summary = (f"Grade Level: {retrieved_profile.get('grade_level', 'Unknown')}\n"
                           f"Strengths: {', '.join(retrieved_profile.get('math_background', {}).get('strengths', []))}\n"
                           f"Challenges: {', '.join(retrieved_profile.get('math_background', {}).get('challenges', []))}\n"
                           f"Learning Preferences: {', '.join(retrieved_profile.get('cognitive_style', {}).get('learning_preferences', []))}")
        return f"Student Background:\n{profile_summary}\n\nQuestion: {question}\n{prefix}\n"
    return f"Question: {question}\n{prefix}\n"

# ----------------------------------------------
# Inference & Logging Logic
# ----------------------------------------------
def generate_cot_and_answer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str):
    """
    Generates a chain-of-thought response + final answer using the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = len(inputs["input_ids"][0])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_length = len(output_ids[0]) - prompt_length

    return output_text, prompt_length, output_length


def extract_final_answer(generated_text: str) -> str:
    """
    Extracts the final answer from the chain-of-thought response.
    """
    answer_patterns = [r"Answer:\s*(.*)", r"Final Answer:\s*(.*)", r"Therefore, the answer is\s*(.*)"]
    for pattern in answer_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return generated_text.strip().split("\n")[-1].strip()


def main():
    args = parse_args()

    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.to(args.device)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")["train"].select(range(args.subset_size))

    print(f"Number of samples to evaluate: {len(ds)}")

    results, total_correct = [], 0
    for idx, sample in enumerate(ds):
        question, gold_solution = sample["question"], sample["answer"]
        retrieved_profile = {"grade_level": "Undergraduate", "math_background": {"strengths": ["Calculus"], "challenges": ["Proofs"]}, "cognitive_style": {"learning_preferences": ["Theory before application"]}}
        prompt = build_cot_prompt(question, args.prompt_prefix, retrieved_profile)

        generated_text, prompt_length, output_length = generate_cot_and_answer(
            model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.device
        )

        predicted_answer = extract_final_answer(generated_text)
        is_correct = predicted_answer.strip() == gold_solution.strip()
        if is_correct:
            total_correct += 1

    print("\nEvaluation Complete.")