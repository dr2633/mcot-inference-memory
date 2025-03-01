## Inference Preference Optimization (IPO): Augmenting GRPO with Memory

This repository explores Memory-Guided Chain-of-Thought (mCoT) reasoning using Group Relative Policy Optimization (GRPO) to optimize inference-time reasoning with user-specific and user-specified memory retrieval with FAISS.

---

## Repository Overview

This repository extends standard Chain-of-Thought (CoT) reasoning by integrating memory and GRPO to personalize reasoning paths based on user preferences and past interactions.

- Memory-Guideded CoT enables models to refine responses over multiple sessions.  
- Inference Preference Optimization (IPO) adapts reasoning preferences dynamically, optimizing inference-time outputs by conditioning GRPO with memory.  
- Customizable User Preferences allows users to specify formatting, reasoning style, and memory.  

---

## Getting Started
This section guides you through setting up the repository and running key scripts.

```python
git clone https://github.com/dr2633/mcot-inference-memory
cd mcot-inference-memory
```

### Install Dependencies
Ensure Python 3.8+ is installed. Then, install the required packages.

```python
 pip install -r requirements.txt
```

### Set Up FAISS for Memory Retrieval
Initialize the FAISS memory store to enable memory retrieval for CoT reasoning.
  
```python
python memory/retrieval_faiss.py --build_index
```


### 3Load and Verify Model
Download and load the Qwen model for Chain-of-Thought reasoning.


Ensure the model and tokenizer are correctly loaded from Hugging Face.

```python
python verify-model.py
```
---

## Running Experiments

Run Baseline Chain-of-Thought (CoT) Evaluation: 

```python
python cot/run_baseline_cot.py --subset_size 50
```

To evaluate standard CoT reasoning on the GSM8K dataset.

### Run Memory-Augmented CoT (mCoT) Evaluation
To evaluate memory-aware reasoning. This version retrieves past responses to guide new reasoning.

```python
python cot/run_memory_cot.py --subset_size 50
```
---

## Reinforcement Learning with GRPO
### Train a GRPO Model with Memory-Augmented Reasoning
Train a model using GRPO with memory augmentation. This optimizes reasoning based on user-specific preferences and past interactions.

```python
python rl/train_rl_memory.py
```

### Evaluate GRPO-Trained Model
To compare pre-trained vs. GRPO-optimized reasoning. This measures accuracy, token efficiency, and personalization fidelity.

```python
python evaluate_rl.py
```

---

### Run Detailed Comparison Baseline CoT vs mCoT
To compare baseline CoT vs. memory-augmented CoT (mCoT). This logs results and highlights improvements from memory-guided reasoning trajectories.

```python
python evaluate_cot_vs_mcot.py
  ```

---

## Running Unit Tests
Before deploying, ensure all tests pass. This runs unit and integration tests for FAISS retrieval, mCoT, and GRPO conditioned with memory (IPO)

```python
pytest tests/
  ```
