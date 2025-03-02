#!/usr/bin/env python
# coding: utf-8
# derek.rosenzweig1@gmail.com

"""
run_baseline_s3.py

Evaluates Qwen on GSM8K dataset across temperature settings.
Collects accuracy, token usage, coherence (perplexity, semantic similarity, entity overlap).
Saves results both locally and to an S3 bucket.
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
import spacy
import boto3  # AWS S3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models for evaluation
nlp = spacy.load("en_core_web_sm")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# AWS S3 Configuration
S3_BUCKET_NAME = "your-s3-bucket-name"
s3_client = boto3.client("s3")

# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Coherence Evaluation of Chain-of-Thought on GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Hugging Face model identifier.")
    parser.add_argument("--subset_size", type=int, default=50, help="Number of examples from GSM8K to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu).")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/qwen_results", help="Local output directory.")
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

def entity_overlap(question, generated_text):
    """Measures overlap of named entities between the question and generated text."""
    q_entities = {ent.text for ent in nlp(question).ents}
    gen_entities = {ent.text for ent in nlp(generated_text).ents}
    return len(q_entities & gen_entities) / max(1, len(q_entities))

def save_to_s3(local_file_path, s3_key):
    """Uploads a file to S3."""
    s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
    print(f"✅ Uploaded {local_file_path} to S3 as {s3_key}")

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
        json_filename = f"results_temp_{temp}_{timestamp}.json"
        local_json_path = os.path.join(args.output_dir, json_filename)

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
            entity_score = entity_overlap(question, generated_text)

            # Store results
            output_tokens_list.append(output_tokens)
            perplexity_list.append(perplexity)
            similarity_list.append(similarity)
            entity_overlap_list.append(entity_score)

            results.append({
                "index": idx, "temperature": temp, "question": question,
                "gold_solution": gold_solution, "generated_text": generated_text,
                "output_tokens": output_tokens, "perplexity": perplexity,
                "semantic_similarity": similarity, "entity_overlap": entity_score
            })

        # Store stats
        token_stats[temp] = {
            "avg_output_tokens": sum(output_tokens_list) / len(output_tokens_list),
            "output_tokens_list": output_tokens_list,
            "perplexity_list": perplexity_list,
            "similarity_list": similarity_list,
            "entity_overlap_list": entity_overlap_list
        }

        # Save locally
        with open(local_json_path, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)
        print(f"Saved {local_json_path} locally.")

        # Upload to S3
        save_to_s3(local_json_path, f"qwen_results/{json_filename}")

    print("Experiment complete. Results saved locally and in S3.")

if __name__ == "__main__":
    main()
