import os
import json
import re
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
    parser = argparse.ArgumentParser(description="Memory-Conditioned Chain-of-Thought Evaluation on GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Hugging Face model identifier.")
    parser.add_argument("--subset_size", type=int, default=50, help="Number of examples from GSM8K to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "data/qwen"),
                        help="Output directory for results.")
    parser.add_argument("--profile_dir", type=str, default=os.path.join(os.getcwd(), "data/profiles"),
                        help="Directory containing student profiles.")
    parser.add_argument("--cot_prompt", type=str,
                        default="Let's reason through this step by step in context of the student's learning needs:",
                        help="Chain-of-thought prompt template.")
    return parser.parse_args()


# ----------------------------------------------
# Evaluation Functions
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
        return float('nan')


def semantic_similarity(generated_text, gold_text, similarity_model):
    """Computes cosine similarity between generated and gold text embeddings."""
    try:
        gen_embedding = similarity_model.encode([generated_text])
        gold_embedding = similarity_model.encode([gold_text])
        return cosine_similarity(gen_embedding, gold_embedding)[0][0]
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return float('nan')


def entity_overlap(question, generated_text, nlp):
    """Measures overlap of named entities between the question and generated text."""
    try:
        q_entities = {ent.text.lower() for ent in nlp(question).ents}
        gen_entities = {ent.text.lower() for ent in nlp(generated_text).ents}
        if not q_entities:
            return 0
        return len(q_entities & gen_entities) / len(q_entities)
    except Exception as e:
        print(f"Error calculating entity overlap: {e}")
        return float('nan')


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


def measure_profile_alignment(profile, generated_text):
    """Measure how well the explanation aligns with student profile needs."""
    alignment_score = 0
    text_lower = generated_text.lower()

    # Check for grade-appropriate vocabulary
    grade_level = profile.get('grade_level', '').lower()
    if 'elementary' in grade_level or '3rd grade' in grade_level or '4th grade' in grade_level:
        # Simple vocabulary check for elementary level
        complex_terms = ['derivative', 'integral', 'quadratic', 'polynomial', 'coefficient']
        simple_explanations = ['step by step', 'let me show', 'easy way', 'simple']

        # Penalize complex terms for elementary students
        alignment_score -= sum(term in text_lower for term in complex_terms) * 0.2
        # Reward simple explanations
        alignment_score += sum(phrase in text_lower for phrase in simple_explanations) * 0.1

    elif 'middle school' in grade_level or '7th grade' in grade_level or '8th grade' in grade_level:
        # Middle-school appropriate checks
        alignment_score += 0.1 if 'let\'s break this down' in text_lower else 0

    elif 'high school' in grade_level or '10th grade' in grade_level:
        # High-school appropriate checks
        alignment_score += 0.1 if any(term in text_lower for term in ['formula', 'equation', 'solve for']) else 0

    # Check for addressing specific strengths mentioned in profile
    strengths = profile.get('math_background', {}).get('strengths', [])
    for strength in strengths:
        if strength.lower() in text_lower:
            alignment_score += 0.15

    # Check for addressing specific challenges mentioned in profile
    challenges = profile.get('math_background', {}).get('challenges', [])
    for challenge in challenges:
        if challenge.lower() in text_lower and (
                'here\'s how to approach' in text_lower or 'let me explain' in text_lower):
            alignment_score += 0.2

    # Check for matching learning preferences
    preferences = profile.get('cognitive_style', {}).get('learning_preferences', [])
    preference_indicators = {
        'visual': ['let me draw', 'picture', 'diagram', 'visualize', 'imagine'],
        'verbal': ['let me explain', 'in words', 'verbally'],
        'example-based': ['for example', 'here\'s an example', 'similar problem'],
        'step-by-step': ['step by step', 'first', 'second', 'next', 'finally'],
        'conceptual': ['concept', 'understand why', 'the idea is', 'intuition']
    }

    for pref in preferences:
        pref_lower = pref.lower()
        for key, indicators in preference_indicators.items():
            if key in pref_lower:
                score_increase = sum(ind in text_lower for ind in indicators) * 0.15
                alignment_score += score_increase

    # Normalize to a 0-1 scale
    return min(max(0.3 + alignment_score, 0), 1)


# ----------------------------------------------
# Profile Loading and FAISS Setup
# ----------------------------------------------
def load_student_profiles(profile_dir):
    """Load student profiles from JSON files in the specified directory."""
    student_profiles = []
    profile_texts = []

    try:
        if not os.path.exists(profile_dir):
            print(f"Profile directory {profile_dir} does not exist.")

            # Create a default set of profiles if directory doesn't exist
            print("Creating default student profiles...")
            student_profiles = create_default_profiles()
            profile_texts = [get_profile_text(p) for p in student_profiles]
            return student_profiles, profile_texts

        for filename in os.listdir(profile_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(profile_dir, filename), "r", encoding="utf-8") as f:
                        profile = json.load(f)
                        profile_text = get_profile_text(profile)
                        student_profiles.append(profile)
                        profile_texts.append(profile_text)
                except Exception as e:
                    print(f"Error loading profile {filename}: {e}")
                    continue

        if not student_profiles:
            print("No valid profiles found. Creating default profiles.")
            student_profiles = create_default_profiles()
            profile_texts = [get_profile_text(p) for p in student_profiles]

    except Exception as e:
        print(f"Error loading profiles: {e}")
        print("Creating default student profiles...")
        student_profiles = create_default_profiles()
        profile_texts = [get_profile_text(p) for p in student_profiles]

    return student_profiles, profile_texts


def get_profile_text(profile):
    """Format a profile as text for embedding."""
    try:
        return (
            f"{profile['grade_level']} student with strengths in {', '.join(profile['math_background']['strengths'])} "
            f"but challenges with {', '.join(profile['math_background']['challenges'])}. "
            f"Prefers {', '.join(profile['cognitive_style']['learning_preferences'])}."
        )
    except KeyError:
        # Handle missing fields gracefully
        parts = []
        if 'grade_level' in profile:
            parts.append(f"{profile['grade_level']} student")

        if 'math_background' in profile:
            bg = profile['math_background']
            if 'strengths' in bg and bg['strengths']:
                parts.append(f"with strengths in {', '.join(bg['strengths'])}")
            if 'challenges' in bg and bg['challenges']:
                parts.append(f"but challenges with {', '.join(bg['challenges'])}")

        if 'cognitive_style' in profile and 'learning_preferences' in profile['cognitive_style']:
            parts.append(f"Prefers {', '.join(profile['cognitive_style']['learning_preferences'])}")

        return " ".join(parts) or "Student profile"


def create_default_profiles():
    """Create a set of default student profiles if none are available."""
    return [
        {
            "profile_id": "elementary_visual",
            "grade_level": "4th Grade",
            "math_background": {
                "strengths": ["basic arithmetic", "pattern recognition"],
                "challenges": ["word problems", "multi-step problems"]
            },
            "cognitive_style": {
                "learning_preferences": ["visual learning", "concrete examples"]
            }
        },
        {
            "profile_id": "middle_school_conceptual",
            "grade_level": "7th Grade",
            "math_background": {
                "strengths": ["fractions", "decimals", "percentages"],
                "challenges": ["algebra", "abstract concepts"]
            },
            "cognitive_style": {
                "learning_preferences": ["conceptual understanding", "step-by-step instructions"]
            }
        },
        {
            "profile_id": "high_school_advanced",
            "grade_level": "10th Grade",
            "math_background": {
                "strengths": ["algebra", "geometry", "equations"],
                "challenges": ["word problems", "probability"]
            },
            "cognitive_style": {
                "learning_preferences": ["formula-based", "example-based learning"]
            }
        }
    ]


def setup_faiss_index(profile_embeddings):
    """Create and populate a FAISS index for profile retrieval."""
    try:
        dimension = profile_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(profile_embeddings).astype('float32'))
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        # Create a simple fallback index
        dimension = profile_embeddings.shape[1] if hasattr(profile_embeddings, 'shape') else 384  # Default for MiniLM
        index = faiss.IndexFlatL2(dimension)
        if hasattr(profile_embeddings, 'shape'):
            index.add(np.array(profile_embeddings).astype('float32'))
        return index


def retrieve_student_profile(question, index, profile_model, student_profiles):
    """Retrieve the most relevant student profile for a given question."""
    try:
        query_embedding = profile_model.encode([question])
        D, I = index.search(np.array(query_embedding).astype('float32'), k=1)
        return student_profiles[I[0][0]], D[0][0]
    except Exception as e:
        print(f"Error retrieving profile: {e}")
        # Return a default profile
        return student_profiles[0] if student_profiles else create_default_profiles()[0], float('inf')


# ----------------------------------------------
# Main Evaluation Function
# ----------------------------------------------
def main():
    args = parse_args()

    print("Loading NLP and embedding models...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading Spacy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    profile_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loading LLM {args.model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        )
        model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading student profiles...")
    student_profiles, profile_texts = load_student_profiles(args.profile_dir)
    print(f"Loaded {len(student_profiles)} student profiles")

    print("Building FAISS index...")
    profile_embeddings = profile_model.encode(profile_texts)
    index = setup_faiss_index(profile_embeddings)

    print("Loading GSM8K dataset...")
    try:
        ds_full = load_dataset("openai/gsm8k", "main")
        ds = ds_full["train"].select(range(min(args.subset_size, len(ds_full["train"]))))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    # temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1]
    temperatures = [0.0, 0.7]

    # Stats storage
    all_results = {}
    token_stats = {}  # For grade level and temperature analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for temp in temperatures:
        print(f"\nEvaluating at temperature {temp} ...")
        results = []

        # Initialize stats for this temperature
        token_stats[temp] = {}

        for idx, sample in enumerate(ds):
            try:
                question, gold_solution = sample["question"], sample["answer"].strip().lower()

                # 1. Retrieve relevant student profile
                student_profile, retrieval_distance = retrieve_student_profile(
                    question, index, profile_model, student_profiles
                )

                grade_level = student_profile.get('grade_level', 'Unknown')
                if grade_level not in token_stats[temp]:
                    token_stats[temp][grade_level] = {
                        "output_tokens_list": [],
                        "perplexity_list": [],
                        "similarity_list": [],
                        "alignment_list": [],
                        "accuracy_list": [],
                        "steps_list": []
                    }

                # Get profile description
                profile_description = get_profile_text(student_profile)

                # 2. Create memory-conditioned prompt
                prompt = f"You are tutoring a {profile_description}. {args.cot_prompt} {question}"

                # 3. Generate response
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(temp > 0),
                        temperature=temp,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Process response
                generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                output_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]

                # 4. Calculate evaluation metrics
                extracted_answer = extract_answer(generated_text)
                is_correct = check_answer_correctness(extracted_answer, gold_solution)
                reasoning_steps = count_reasoning_steps(generated_text)
                perplexity = calculate_perplexity(generated_text, model, tokenizer, args.device)
                similarity = semantic_similarity(generated_text, gold_solution, similarity_model)
                profile_alignment = measure_profile_alignment(student_profile, generated_text)

                # Store metrics
                token_stats[temp][grade_level]["output_tokens_list"].append(output_tokens)
                token_stats[temp][grade_level]["perplexity_list"].append(perplexity)
                token_stats[temp][grade_level]["similarity_list"].append(similarity)
                token_stats[temp][grade_level]["alignment_list"].append(profile_alignment)
                token_stats[temp][grade_level]["accuracy_list"].append(1 if is_correct else 0)
                token_stats[temp][grade_level]["steps_list"].append(reasoning_steps)

                # 5. Store detailed results
                results.append({
                    "index": idx,
                    "temperature": temp,
                    "question": question,
                    "gold_solution": gold_solution,
                    "retrieved_profile": profile_description,
                    "retrieval_distance": float(retrieval_distance),
                    "generated_text": generated_text,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "output_tokens": output_tokens,
                    "reasoning_steps": reasoning_steps,
                    "perplexity": float(perplexity) if not np.isnan(perplexity) else None,
                    "semantic_similarity": float(similarity) if not np.isnan(similarity) else None,
                    "profile_alignment": float(profile_alignment)
                })

                if (idx + 1) % 5 == 0:
                    print(f"Processed {idx + 1}/{len(ds)} examples")

            except Exception as e:
                print(f"Error processing example {idx} at temperature {temp}: {e}")
                continue

        # Store results by temperature
        all_results[temp] = results

        # Save individual temperature results
        json_filename = os.path.join(args.output_dir, f"mcot_results_temp_{temp}_{timestamp}.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"temperature": temp, "results": results}, f, indent=2)

        print(f"Results saved to {json_filename}")

        # Calculate aggregated metrics for this temperature
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct_count / len(results) if results else 0
        avg_profile_alignment = sum(r.get("profile_alignment", 0) for r in results) / len(results) if results else 0

        print(f"Temperature {temp} | Accuracy: {accuracy:.4f} | Avg Alignment: {avg_profile_alignment:.4f}")

    # Save summary statistics
    summary_stats = {}
    for temp in temperatures:
        temp_stats = {"overall": {}, "by_grade_level": {}}

        # Calculate overall metrics
        all_correct = sum(1 for r in all_results[temp] if r.get("is_correct", False))
        total = len(all_results[temp])

        temp_stats["overall"] = {
            "accuracy": all_correct / total if total else 0,
            "avg_tokens": sum(r.get("output_tokens", 0) for r in all_results[temp]) / total if total else 0,
            "avg_steps": sum(r.get("reasoning_steps", 0) for r in all_results[temp]) / total if total else 0,
            "avg_alignment": sum(r.get("profile_alignment", 0) for r in all_results[temp]) / total if total else 0
        }

        # Calculate per grade level metrics
        for grade, metrics in token_stats[temp].items():
            if metrics["accuracy_list"]:
                temp_stats["by_grade_level"][grade] = {
                    "accuracy": sum(metrics["accuracy_list"]) / len(metrics["accuracy_list"]),
                    "avg_tokens": sum(metrics["output_tokens_list"]) / len(metrics["output_tokens_list"]),
                    "avg_steps": sum(metrics["steps_list"]) / len(metrics["steps_list"]),
                    "avg_alignment": sum(metrics["alignment_list"]) / len(metrics["alignment_list"])
                }

        summary_stats[temp] = temp_stats

    # Save summary stats
    summary_file = os.path.join(args.output_dir, f"mcot_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    # Create visualizations
    create_visualizations(token_stats, args.output_dir, timestamp)

    print(f"All results saved to {args.output_dir}")
    print("\n==== Summary of Results ====")
    for temp in temperatures:
        print(f"Temperature {temp}:")
        print(f"  Overall Accuracy: {summary_stats[temp]['overall']['accuracy']:.4f}")
        print(f"  Average Profile Alignment: {summary_stats[temp]['overall']['avg_alignment']:.4f}")
        print(f"  Average Reasoning Steps: {summary_stats[temp]['overall']['avg_steps']:.2f}")
        print("  By Grade Level:")
        for grade, stats in summary_stats[temp]["by_grade_level"].items():
            print(f"    {grade}: Accuracy={stats['accuracy']:.4f}, Alignment={stats['avg_alignment']:.4f}")


def create_visualizations(token_stats, output_dir, timestamp):
    """Create visualization plots from the evaluation results."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for temp, grade_data in token_stats.items():
        for grade, metrics in grade_data.items():
            for i in range(len(metrics["output_tokens_list"])):
                try:
                    entry = {
                        "Temperature": temp,
                        "Grade Level": grade,
                        "Output Tokens": metrics["output_tokens_list"][i],
                        "Perplexity": metrics["perplexity_list"][i],
                        "Semantic Similarity": metrics["similarity_list"][i],
                        "Profile Alignment": metrics["alignment_list"][i],
                        "Is Correct": metrics["accuracy_list"][i],
                        "Reasoning Steps": metrics["steps_list"][i]
                    }
                    plot_data.append(entry)
                except IndexError:
                    continue

    if not plot_data:
        print("No data available for visualization")
        return

    df = pd.DataFrame(plot_data)
    sns.set(style="whitegrid")

    # 1. Violin Plot: Output Tokens by Grade Level and Temperature
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x="Temperature", y="Output Tokens", hue="Grade Level", data=df,
                        palette="muted", split=True)
    plt.title("Output Tokens by Grade Level and Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Number of Output Tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"output_tokens_violin_{timestamp}.png"))
    plt.close()

    # 2. Bar Plot: Accuracy by Grade Level and Temperature
    plt.figure(figsize=(12, 8))
    accuracy_df = df.groupby(["Temperature", "Grade Level"])["Is Correct"].mean().reset_index()
    sns.barplot(x="Temperature", y="Is Correct", hue="Grade Level", data=accuracy_df, palette="muted")
    plt.title("Accuracy by Grade Level and Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"accuracy_bar_{timestamp}.png"))
    plt.close()

    # 3. Heatmap: Profile Alignment by Grade Level and Temperature
    plt.figure(figsize=(10, 8))
    alignment_df = df.groupby(["Temperature", "Grade Level"])["Profile Alignment"].mean().reset_index()
    alignment_pivot = alignment_df.pivot(index="Grade Level", columns="Temperature", values="Profile Alignment")
    sns.heatmap(alignment_pivot, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Profile Alignment by Grade Level and Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"alignment_heatmap_{timestamp}.png"))
    plt.close()

    # 4. Line Plot: Reasoning Steps by Temperature
    plt.figure(figsize=(10, 6))
    steps_df = df.groupby(["Temperature", "Grade Level"])["Reasoning Steps"].mean().reset_index()
    sns.lineplot(x="Temperature", y="Reasoning Steps", hue="Grade Level",
                 data=steps_df, markers=True, dashes=False)
    plt.title("Average Reasoning Steps by Temperature and Grade Level")
    plt.xlabel("Temperature")
    plt.ylabel("Average Number of Reasoning Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"reasoning_steps_line_{timestamp}.png"))
    plt.close()

    # 5. Scatter Plot: Profile Alignment vs Accuracy
    plt.figure(figsize=(10, 6))
    alignment_accuracy_df = df.groupby(["Temperature", "Grade Level"])[
        ["Profile Alignment", "Is Correct"]].mean().reset_index()
    sns.scatterplot(x="Profile Alignment", y="Is Correct", hue="Grade Level",
                    size="Temperature", sizes=(50, 200), data=alignment_accuracy_df)
    plt.title("Profile Alignment vs Accuracy")
    plt.xlabel("Profile Alignment Score")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"alignment_vs_accuracy_{timestamp}.png"))
    plt.close()


if __name__ == "__main__":
    main()