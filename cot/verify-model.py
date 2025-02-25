from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen-7B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Qwen Model and Tokenizer loaded successfully!")
