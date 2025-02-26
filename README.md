## Memory-Guided Chain-of-Thought (mCoT) with GRPO for Inference Preference Optimization (IPO)

This repository explores Memory-Guided Chain-of-Thought (mCoT) reasoning using Group Relative Policy Optimization (GRPO) to optimize inference-time reasoning with user-specific memory.

---

## Repository Overview

This repository extends standard Chain-of-Thought (CoT) reasoning by integrating memory and GRPO to personalize reasoning paths based on user preferences and past interactions.

- Memory-Guideded CoT → Enables models to refine responses over multiple sessions.  
- Inference Preference Optimization (IPO) → Adapts reasoning preferences dynamically, optimizing inference-time outputs by conditioning GRPO with memory.  
- Customizable User Preferences → Users can define formatting, reasoning style, and memory persistence.  

---

## Getting Started
This section guides you through setting up the repository and running key scripts.

```python
git clone https://github.com/dr2633/mCoT-GRPO-IPO
cd mCoT-GRPO-IPO
```

### Install Dependencies
Ensure Python 3.8+ is installed. Then, install the required packages.

```python
 pip install -r requirements.txt
```

### Set Up FAISS for Memory Retrieval
Initialize the **FAISS memory store** to enable memory retrieval for CoT reasoning.
  
```python
python memory/retrieval_faiss.py --build_index
```


### 3Load and Verify Model
Download and load the Qwen model for Chain-of-Thought reasoning.

  
```python
python verify-model.py
```

This will **ensure the model and tokenizer** are correctly loaded from Hugging Face.

---

## Running Experiments
### Run Baseline Chain-of-Thought (CoT) Evaluation

```python
python cot/run_baseline_cot.py --subset_size 50
```

To evaluate **standard CoT reasoning** on the **GSM8K dataset**.

### Run Memory-Augmented CoT (mCoT) Evaluation
To evaluate **memory-aware reasoning**. This version **retrieves past responses** to guide new reasoning.

```python
python cot/run_memory_cot.py --subset_size 50
```

---

## Reinforcement Learning with GRPO
### Train a GRPO Model with Memory-Augmented Reasoning
Train a model using **GRPO with memory augmentation**. This optimizes reasoning **based on user-specific preferences** and **past interactions**.

```python
python rl/train_rl_memory.py
```

### Evaluate GRPO-Trained Model
To compare **pre-trained vs. GRPO-optimized reasoning**. This measures **accuracy, token efficiency, and personalization fidelity**.

```python
python evaluate_rl.py
```

---

### Run Detailed Comparison Baseline CoT vs mCoT
To compare **baseline CoT vs. memory-augmented CoT (mCoT)**. This logs results and **highlights improvements from memory-aware reasoning**.

```python
python evaluate_cot_vs_mcot.py
  ```

---

## Running Unit Tests
Before deploying, ensure all tests pass. This runs **unit and integration tests** for:
- **Memory Retrieval**
- **Chain-of-Thought Logic**
- **GRPO Model Performance**

```python
pytest tests/
  ```

---

## Configurations
Modify these config files for customization:
- **Memory Configuration** → Adjust FAISS retrieval parameters.
- **GRPO Configuration** → Modify reinforcement learning settings.
- **User Preferences** → Customize reasoning styles.

---

## Next Steps
1. **Run model training** to refine reasoning through **GRPO fine-tuning**.  
2. **Experiment with user preferences** to personalize model outputs.  
3. **Adjust FAISS retrieval and re-run evaluations**.  
4. **Fine-tune GRPO hyperparameters to improve learning efficiency**.  

---

