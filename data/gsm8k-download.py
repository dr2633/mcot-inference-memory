from datasets import load_dataset
import json
import os

# Get the absolute path of the project's data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load GSM8K dataset with the "main" configuration
dataset = load_dataset("openai/gsm8k", "main")

# Define paths for local storage
train_path = os.path.join(DATA_DIR, "gsm8k_train.json")
test_path = os.path.join(DATA_DIR, "gsm8k_test.json")

# Convert dataset to a list of dictionaries (required for JSON serialization)
train_data = dataset["train"].to_list()  # Convert Dataset object to a list
test_data = dataset["test"].to_list()    # Convert Dataset object to a list

# Save train and test splits locally
with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

print(f"GSM8K dataset successfully downloaded and stored in {DATA_DIR}")
