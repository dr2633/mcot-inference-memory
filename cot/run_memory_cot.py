# `run_memory_cot.py`
#!/usr/bin/env python
# coding: utf-8

"""
run_memory_cot.py

Demonstrates a Memory-Augmented Chain-of-Thought (mCoT) approach by retrieving
relevant past interactions from a FAISS-based memory store and incorporating
them into the prompt to shape Qwen's reasoning on GSM8K.
"""

import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your custom memory store (assumes memory_store.py is in the same directory)
# from memory_store import MemoryStore

# ----------------------------------------------
# Argument Parsing
# ----------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Memory-Augmented CoT on GSM8K with Qwen + FAISS retrieval")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="Hugging Face model identifier (Qwen) or local path."
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
        help="Max tokens to generate for chain-of-thought + final answer."
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
        help="Device to run inference on (cuda, cpu, or mps)."
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="Let's recall past steps and reason carefully step-by-step.",
        help="Prompt prefix to encourage chain-of-thought reasoning."
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="faiss_index",
        help="Path or directory where the FAISS index and related files are stored."
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="user_001",
        help="Identifier for user memory retrieval (if relevant)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="memory_cot_results.json",
        help="Where to store the JSON log of results."
    )
    return parser.parse_args()

# ----------------------------------------------
# Prompt Construction
# ----------------------------------------------
def build_memory_prompt(question: str, prefix: str, retrieved_context: str) -> str:
    """
    Build a memory-augmented CoT prompt by incorporating retrieved memory/context
    before the chain-of-thought prefix and question.
    """
    # You can revise the structure as needed.
    # Example structure: [Retrieved context] + [Prefix] + "Question: " + [question]
    return (
        f"Relevant past context:\n{retrieved_context}\n\n"
        f"{prefix}\n\n"
        f"Question: {question}\n"
    )

# ----------------------------------------------
# Inference Logic
# ----------------------------------------------
def generate_memory_cot(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str
):
    """
    Generates chain-of-thought + final answer using the model, incorporating
    the memory-augmented prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_final_answer(generated_text: str) -> str:
    """
    Extract final answer from the chain-of-thought. Looks for an 'Answer:' marker or returns last line.
    """
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    # fallback to entire text
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
    ds = ds_full["train"]

    if args.subset_size > 0 and args.subset_size < len(ds):
        ds = ds.select(range(args.subset_size))

    print(f"Number of samples to evaluate: {len(ds)}")

    # 3. Initialize the memory store (FAISS-based)
    # memory_store = MemoryStore(index_path=args.faiss_index_path)
    # memory_store.load_index()  # hypothetical method to load an existing FAISS index

    # 4. Evaluation placeholders
    results = []
    total_correct = 0

    # 5. Iterate over dataset
    for idx, sample in enumerate(ds):
        question = sample["question"]
        gold_solution = sample["answer"]

        # Retrieve memory for the user or question
        # retrieved_memory = memory_store.retrieve_top_k(
        #     query_text=question,
        #     user_id=args.user_id,
        #     k=2
        # )
        # If you're returning text chunks, combine them into a single string:
        # retrieved_context = "\n".join([mem["content"] for mem in retrieved_memory])

        # For demonstration (FAISS not shown), we provide a stub:
        retrieved_context = "No relevant past context found."  # Or from memory retrieval

        # Build memory-augmented prompt
        prompt = build_memory_prompt(question, args.prompt_prefix, retrieved_context)

        # Generate chain-of-thought with memory
        generated_text = generate_memory_cot(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )

        # Extract final answer
        predicted_answer = extract_final_answer(generated_text)

        # Token length stats
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(generated_text)["input_ids"]

        # Naive correctness check
        is_correct = (predicted_answer.strip() == gold_solution.strip())
        if is_correct:
            total_correct += 1

        # Log the results
        result_entry = {
            "index": idx,
            "question": question,
            "gold_solution": gold_solution,
            "retrieved_context": retrieved_context,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "prompt_length": len(prompt_tokens),
            "output_length": len(output_tokens),
        }
        results.append(result_entry)

        # Print partial progress
        if (idx + 1) % 5 == 0:
            print(f"[{idx+1}/{len(ds)}] Current accuracy: {total_correct/(idx+1):.2%}")

    # 6. Compute final metrics
    overall_accuracy = total_correct / len(ds) if len(ds) else 0
    avg_prompt_len = sum(r["prompt_length"] for r in results) / max(len(results), 1)
    avg_output_len = sum(r["output_length"] for r in results) / max(len(results), 1)

    print("\nEvaluation Complete.")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print(f"Average Prompt Token Length: {avg_prompt_len:.2f}")
    print(f"Average Output Token Length: {avg_output_len:.2f}")

    # 7. Save results
    import json
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "avg_prompt_length": avg_prompt_len,
            "avg_output_length": avg_output_len,
            "results": results
        }, f, indent=2)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
