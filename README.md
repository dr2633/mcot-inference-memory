## Inference Preference Optimization (IPO): Augmenting GRPO with Memory

## Overview
Inference Preference Optimization (IPO) enhances reasoning in LLMs by integrating memory retrieval into Group Relative Policy Optimization (GRPO), enabling models to adapt responses dynamically based on a user's prior learning.

This repository first evaluates memory-conditioned Chain-of-Thought (mCoT) reasoning in LLMs on the GSM8K dataset using memory retrieval with FAISS. mCoT serves as a critical validation step for IPO by demonstrating how consolidating prior interactions and retrieving relevant user-specific memory can personalize model reasoning. Once validated, GRPO further optimizes inference paths by conditioning reinforcement learning on retrieved memory, forming the basis of IPO.

### Key Features
- **FAISS-based retrieval** to match questions with relevant student profiles
- **Memory-conditioned CoT (mCoT)** for dynamically refined responses, validating memory retrieval effectiveness
- **Inference Preference Optimization (IPO):** GRPO conditioned with user memory to personalize reasoning sequences
- **Textbook-based evaluation** for GRPO, tracking reasoning adaptation based on retrieved user progress

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
Or manually install:
```bash
pip install torch transformers datasets faiss-cpu sentence-transformers matplotlib seaborn pandas spacy
python -m spacy download en_core_web_sm
```

### 2. Clone Repository
```bash
git clone https://github.com/your-repo/mCoT-GRPO-IPO.git
cd mCoT-GRPO-IPO
```

### 3. Connecting to a Lambda Instance

#### **1. Set Up SSH Access**
Generate an SSH key pair if you do not have one:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/lambda_key -C "your-email@example.com"
```
Ensure the key has the correct permissions:
```bash
chmod 600 ~/.ssh/lambda_key
```

#### **2. Add Your Public Key to Lambda Cloud**
1. Log in to [Lambda Cloud](https://lambdalabs.com/cloud).
2. Go to **SSH Keys** under **Settings**.
3. Click **Add SSH Key**.
4. Copy the public key:
   ```bash
   cat ~/.ssh/lambda_key.pub
   ```

#### **3. Connect to Your Instance**
```bash
ssh -i ~/.ssh/lambda_key ubuntu@<your-instance-ip>
```

Verify that the GPU is detected:
```bash
nvidia-smi
```

## Usage

### 1. Run Memory-Conditioned CoT Evaluation
```bash
cd cot 
python run_baseline_cot.py --model_name Qwen/Qwen-7B --subset_size 50 --max_new_tokens 256 --device cuda
```

This script runs Chain-of-Thought (CoT) reasoning on a subset of GSM8K. You can swap models as needed.


To validate FAISS-based memory retrieval, use the following script:

```bash
python run_baseline_faiss.py --model_name Qwen/Qwen-7B --subset_size 50 --max_new_tokens 256 --device cuda
```

This script:
- Retrieves the most relevant student profile using FAISS
- Generates explanations based on the retrieved profile
- Evaluates results across different temperatures

Example result format:
```json
{
  "index": 5,
  "temperature": 0.6,
  "question": "If a train travels 60 mph for 3 hours, how far does it go?",
  "gold_solution": "180 miles.",
  "retrieved_profile": "5th grade",
  "generated_text": "The train goes 180 miles because distance = speed × time.",
  "output_tokens": 42,
  "perplexity": 12.3,
  "semantic_similarity": 0.85
}
```


### 2. Validate Outputs
Results are stored in:
```
data/qwen/results_temp_<temperature>_<timestamp>.json
```
To inspect:
```bash
cat data/qwen/results_temp_0.6_<timestamp>.json | jq .
```

## Reinforcement Learning with GRPO

### 1. Train GRPO with Memory-Augmented Reasoning
Train a model using GRPO to optimize reasoning paths based on memory retrieval.
```bash
python rl/train_rl_memory.py
```

### 2. Evaluate GRPO-Trained Model
Compare pre-trained vs. GRPO-optimized reasoning:
```bash
python evaluate_rl.py
```


