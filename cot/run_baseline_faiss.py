import os
import json
import faiss
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


# Load models for evaluation
nlp = spacy.load("en_core_web_sm")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
profile_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load student profiles from JSON files
def load_student_profiles(profile_dir):
    student_profiles = []
    profile_texts = []
    for filename in os.listdir(profile_dir):
        if filename.endswith(".json"):
            with open(os.path.join(profile_dir, filename), "r", encoding="utf-8") as f:
                profile = json.load(f)
                profile_text = (
                    f"{profile['grade_level']} student with strengths in {', '.join(profile['math_background']['strengths'])} "
                    f"but challenges with {', '.join(profile['math_background']['challenges'])}. "
                    f"Prefers {', '.join(profile['cognitive_style']['learning_preferences'])}."
                )
                student_profiles.append(profile)
                profile_texts.append(profile_text)
    return student_profiles, profile_texts


profile_dir = "/Users/derekrosenzweig/Documents/GitHub/mCoT-GRPO-IPO/data/"
student_profiles, profile_texts = load_student_profiles(profile_dir)

# Encode student profiles and build FAISS index
profile_embeddings = profile_model.encode(profile_texts)
index = faiss.IndexFlatL2(profile_embeddings.shape[1])
index.add(np.array(profile_embeddings))


# Function to retrieve the most relevant student profile
def retrieve_student_profile(question):
    query_embedding = profile_model.encode([question])
    D, I = index.search(np.array(query_embedding), k=1)
    return student_profiles[I[0][0]]


# Modify evaluation loop
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

        for idx, sample in enumerate(ds):
            question, gold_solution = sample["question"], sample["answer"].strip().lower()

            # Retrieve relevant student profile
            student_profile = retrieve_student_profile(question)
            profile_description = (
                f"{student_profile['grade_level']} student with strengths in {', '.join(student_profile['math_background']['strengths'])} "
                f"but challenges with {', '.join(student_profile['math_background']['challenges'])}. "
                f"Prefers {', '.join(student_profile['cognitive_style']['learning_preferences'])}."
            )

            # Modify prompt with retrieved profile
            prompt = f"You are tutoring a {profile_description}. Let's reason through this step by step in context of the student's learning needs: {question}"

            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                            temperature=temp)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Store results
            results.append({
                "index": idx, "temperature": temp, "question": question,
                "gold_solution": gold_solution, "retrieved_profile": profile_description,
                "generated_text": generated_text
            })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(args.output_dir, f"results_temp_{temp}_{timestamp}.json")

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)

        print(f"Results saved to {json_filename}")

    # Prepare Data for Visualization
    plot_data = []
    for (temp, grade_level), values in token_stats.items():
        for i in range(len(values["output_tokens_list"])):
            plot_data.append({
                "Temperature": temp,
                "Grade Level": grade_level,
                "Output Tokens": values["output_tokens_list"][i]
            })
    df = pd.DataFrame(plot_data)

    # Violin Plot: Output Tokens Across Grade Levels and Temperatures
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Temperature", y="Output Tokens", hue="Grade Level", data=df, palette="coolwarm", split=True)
    plt.title("Reasoning Depth: Output Length Across Grade Levels and Temperatures")
    plt.savefig(os.path.join(args.output_dir, f"output_tokens_violin_{timestamp}.png"))
    plt.show()

    # Scatter Plot: Perplexity vs. Semantic Similarity
    plt.figure(figsize=(10, 6))
    scatter_df = df.groupby(["Temperature", "Grade Level"])[["Perplexity", "Semantic Similarity"]].mean().reset_index()
    sns.scatterplot(x="Perplexity", y="Semantic Similarity", hue="Grade Level", data=scatter_df, palette="viridis",
                    style="Temperature", size="Semantic Similarity")
    plt.title("Fluency vs. Coherence: Perplexity vs. Similarity Across Grade Levels")
    plt.savefig(os.path.join(args.output_dir, f"perplexity_similarity_scatter_{timestamp}.png"))
    plt.show()


if __name__ == "__main__":
    main()