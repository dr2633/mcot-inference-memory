# Memory-Guided Chain-of-Thought (mCoT) with GRPO for Inference Preference Optimization (IPO)

This repository explores how Large Language Models (LLMs) can move beyond single-turn or single-session interactions by integrating memory into their inference-time reasoning processes. We build upon:

- **Chain-of-Thought (CoT)**: step-by-step reasoning  
- **Group Relative Policy Optimization (GRPO)**: a reinforcement learning method without a value model, emphasizing efficiency and scalability  
- **Inference Preference Optimization (IPO)**: our proposed extension of GRPO to dynamically adapt inference-time reasoning with user-specific memory

**Goal:** Enable LLMs to leverage past reasoning trajectories and user feedback to deliver personalized, long-term interactions — particularly for complex tasks like math reasoning, code generation, or long-form problem-solving where user preferences evolve over multiple sessions.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Key Concepts and Framework](#key-concepts-and-framework)  
   1. [Standard Chain-of-Thought (CoT)](#standard-chain-of-thought)  
   2. [Memory-Augmented CoT (mCoT)](#memory-augmented-cot)  
   3. [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization)  
   4. [Inference Preference Optimization (IPO)](#inference-preference-optimization)  
3. [Mathematical Formulations](#mathematical-formulations)  
   1. [CoT Equations](#cot-equations)  
   2. [Memory Integration](#memory-integration)  
   3. [Extending GRPO with Memory (GRPO-M)](#extending-grpo-with-memory-grpo-m)  
   4. [Inference-Time Preference Optimization](#inference-time-preference-optimization)  
4. [Research Questions](#research-questions)  
5. [Benchmarks and Experiments](#benchmarks-and-experiments)  
   1. [TinyZero: Countdown & Multiplication Tasks](#tinyzero-countdown--multiplication-tasks)  
   2. [Memory-Guided Math Formatting](#memory-guided-math-formatting)  
   3. [Evaluation Metrics](#evaluation-metrics)  
6. [Project Roadmap](#project-roadmap)  
7. [Getting Started](#getting-started)  
8. [References](#references)

---

## Motivation


- Traditional Chain-of-Thought prompting treats each user query in isolation. Context is reintroduced manually every session.  
- Memory-augmented reasoning retains and updates context, enabling the model to adapt over multi-session workflows, maintain coherence, and refine responses to user-specific goals.
- Integrating memory can significantly improve personalization, reduce repeated user input, and enable user-driven customization (e.g., consistent formatting for math outputs or code style).

---

## Key Concepts and Framework

### Standard Chain-of-Thought (CoT)

In Chain-of-Thought (CoT) reasoning:

1. The model generates a sequence of intermediate steps $\{s_1, s_2, \dots, s_n\}$.  
2. A final output $O$ is derived from these steps.

This approach enables step-by-step transparency but lacks a built-in memory mechanism for multi-turn personalization.

### Memory-Guided CoT (mCoT)

**Memory-Guided CoT (mCoT)** extends CoT by adding a memory state $M_t$ at each step:

- Tracks user feedback, self-consistency checks, and context across sessions  
- Allows the model to adapt its reasoning based on previously stored interactions (ie. json files)

This leads to persistent, user-tailored reasoning trajectories, rather than single-session solutions driven by generic preferences learned during pretraining. 

### Group Relative Policy Optimization (GRPO)

GRPO (introduced in DeepSeek) is an alternative to PPO:

- **No Value Model**: Reduces overhead by eliminating the value function  
- **Relative Ranking**: Optimizes policies based on relative preferences among generated outputs  
- **Stability**: Utilizes a clipped objective and a KL penalty for stable updates

Originally designed for training-time preference optimization, GRPO can excel at tasks requiring robust reward or preference modeling (e.g., math, code, logic).

### Inference Preference Optimization (IPO)

**Inference Preference Optimization** (IPO) is our proposed extension of GRPO to inference time:

- Dynamically retrieves user memory  
- Updates CoT reasoning with user-specific preferences  
- Continuously adapts to user feedback—no need for a separate fine-tuning step after deployment

By conditioning the GRPO framework on retrieved memory, we can shape and refine model outputs to a specific user, leading to personalized, consistent interactions and planning.

---

## Mathematical Formulations

### Memory-Guided Chain-of-Thought (mCoT)

Standard CoT process:

$$
s_t = f\bigl(s_{t-1}, c_t\bigr), \quad t=1,\dots,n
$$

$$
O = g\bigl(s_1, s_2, \dots, s_n\bigr)
$$

- $s_t$: intermediate reasoning state  
- $c_t$: context at step \(t\)  
- $O$: final output

### Memory Integration

We introduce a memory state $M_t$:

$$
s_t = f\bigl(s_{t-1}, c_t, M_t\bigr), \quad
M_t = h\bigl(M_{t-1}, s_{t-1}, c_t, r_t\bigr)
$$

- $M_t$: memory state at step $t$  
- $h$: memory update function  
- $r_t$: refinement signals (user feedback, self-consistency checks, etc.)

**Memory Decay (Optional):**

$$
M_t = \lambda M_{t-1} + h\bigl(s_{t-1}, c_t, r_t\bigr), \quad 0 < \lambda < 1
$$

### Inference Preference Optimization (IPO)

In **standard GRPO**, we sample outputs and compute rewards. Here, we incorporate **memory**:


1. **Memory-Augmented Policy Distribution**  

   $$
   \pi_\theta(o \mid q, M_u) 
   = \frac{\exp \bigl(A(o, q, M_u)\bigr)}{\sum_{o' \in O} \exp \bigl(A(o', q, M_u)\bigr)}

 $$
   
   where  
   $$
   A(o, q, M_u) = z(o, q) + \lambda \cdot S(o, M_u).
   $$
   
   - $z(o, q)$: z-score normalized reward  
   - $S(o, M_u)$: alignment score with user-specific memory  
   - $\lambda$: scales the memory contribution
 

3. **Memory Retrieval**  

   $$
   M_u = R(q, H_u) = \arg\max_{M' \in H_u} \Phi(q, M')
   $$
   
   where $\Phi$ is a semantic similarity or attention-based function, and $H_u$ is the user’s historical memory bank.
 

4. **Inference Preference Optimization Loss Function**  

   $$
   J_{\text{GRPO-M}}(\theta) = \mathbb{E}\Bigl[\sum_{t=1}^{|o|} \min \bigl(r_t(\theta), \,\text{clip}\bigl(r_t(\theta), 1-\epsilon, 1+\epsilon\bigr)\bigr)\Bigr]
   $$
   
   with:
   $$
   r_t(\theta) \propto \frac{\pi_\theta(o_t \mid q, M_u)}{\pi_\theta^{\text{old}}(o_t \mid q, M_u)} \cdot A_t,
   $$

   where $A_t$ now includes memory-conditioned advantage signals.

### Inference-Time Preference Optimization

Rather than relying on **static** preference models:

- **IPO** updates the policy at **inference** based on newly retrieved memory and user feedback  
- Models can incorporate the user’s evolving preferences (e.g., formatting style for math) in real time

---

## Research Questions

We aim to evaluate the **long-term viability** of memory-conditioned reasoning through the following **benchmarks** and **open questions**:

1. **Storage and Retrieval for Efficient Inference**  
   - **RQ1**: How should long-term memory be structured and retrieved to shape CoT reasoning across sessions?

2. **Benchmarks for Reliability in Long-Form Tasks**  
   - **RQ2**: Does memory-conditioned CoT improve precision and reliability over extended interactions?

3. **User Modifications to Memory**  
   - **RQ3**: How configurable should user memory be? Can users manually add/remove or rank memory chunks?

4. **Avoiding Overfitting (the Role of \(\lambda\))**  
   - **RQ4**: How can we ensure that CoT remains flexible, adapting to user changes instead of overfitting to stale history?

---

## Benchmarks and Experiments

### TinyZero: Countdown & Multiplication Tasks

We build on the [**TinyZero** repository](https://github.com/Jiayi-Pan/TinyZero), a reproduction of DeepSeek R1 Zero:

- **Countdown** and **Multiplication** tasks are used to benchmark emergent reasoning and self-verification  
- We will integrate **mCoT** + **GRPO-M** within these tasks to see how the model handles repeated tasks, user-specific math formatting preferences, and multi-session continuity

### Memory-Guided Math Formatting

A concrete example:

- **User**: “When you show your derivations, please label each step with `(Step #)`, and highlight final results in `$\LaTeX$ bold`.”

Our approach will store this **preference** in memory and retrieve it for subsequent tasks, automatically formatting math outputs to the user’s desired style.

### Evaluation Metrics

- **Accuracy / Task Success**: Standard correctness metrics for math or logic tasks  
- **Personalization Fidelity**: The extent to which the model adheres to user-defined preferences (e.g., formatting style)  
- **Continuity**: Performance on multi-session tasks without context reintroduction from the user  
- **Overfitting Check**: A measure of the model’s flexibility—can it handle shifts in user style or new tasks?

---

## Project Roadmap

1. **Initial Setup**  
   - Clone [TinyZero](https://github.com/Jiayi-Pan/TinyZero)  
   - Implement basic CoT reasoning on top of TinyZero’s tasks

2. **Add Memory Component**  
   - Create a simple memory store for user preferences and past reasoning states  
   - Implement a retrieval mechanism (e.g., semantic search or approximate nearest neighbor)

3. **Integrate GRPO-M**  
   - Adapt GRPO to incorporate memory-conditioned rewards and advantage functions at inference time  
   - Add hyperparameters for \(\lambda\) (memory decay) and the retrieval scope

4. **Run Benchmarks**  
   - Evaluate on countdown and multiplication tasks  
   - Introduce user-driven style preferences for math outputs  
   - Measure performance improvements, continuity, and user satisfaction

5. **Refine & Extend**  
   - Experiment with different memory architectures (e.g., LSTM memory, key-value store, or attention-based)  
   - Investigate memory curation and user controls for adding/removing memory entries  
   - Explore scaling to more complex tasks or larger LMs

---

## Getting Started

1. **Clone this repository** (or your fork):
   ```bash
   git clone https://github.com/<YourUsername>/mCoT-GRPO-IPO.git
   cd mCoT-GRPO-IPO

2. **Getting Started**

  - Python 3.8+ recommended
  - PyTorch, Transformers, plus RL libraries

   ```bash
    pip install -r requirements.txt
   ```

3. **Set Up TinyZero**

    - Follow instructions in TinyZero to install or incorporate the countdown/multiplication tasks.
    - Link or copy relevant modules into this project for integrated testing.

4. **Run a Example** 

```bash
 python run_experiment.py --task countdown --use_memory True --lambda_decay 0.9
  ```

This script demonstrates a sample pipeline with memory-enabled CoT + GRPO-M. Adjust hyperparameters as needed.

5. **Customize Memory** 
   - In `config/memory_config.yaml`, set retrieval parameters, memory size, or decay rates.
   - Adjust user preference stubs in `scripts/user_preference.py` to see how the model formats outputs with user-specific guidelines.


**Repo Structure**

# mCoT-GRPO-IPO Repository Structure

This document outlines the **high-level directory structure** for the **Memory-Guided Chain-of-Thought (mCoT)** project. Each folder and script is briefly described in terms of **purpose** and **intended usage**. Additionally, **key questions** or **next-step clarifications** are presented as bullet points to guide future development.

---

## **1. Directory Overview**

```plaintext
mCoT-GRPO-IPO/
├── config/
│   ├── memory_config.yaml
│   ├── grpo_config.yaml
│   └── experiment_config.yaml
├── data/
│   ├── sample_memory.json
│   └── gsm8k_results/
├── memory/
│   ├── memory_store.py
│   └── retrieval_faiss.py
├── cot/
│   ├── run_baseline_cot.py
│   ├── run_memory_cot.py
│   └── prompts/
├── rl/
│   ├── train_rl_memory.py
│   ├── grpo_utilities.py
│   └── reward_functions.py
├── scripts/
│   ├── evaluate_cot_vs_mcot.py
│   ├── evaluate_rl.py
│   └── user_preference.py
├── tests/
│   ├── test_memory.py
│   ├── test_cot.py
│   ├── test_rl.py
│   └── test_integration.py
├── examples/
│   ├── gsm8k_demo.ipynb
│   ├── textbook_demo.ipynb
│   └── ...
├── README.md
├── requirements.txt
└── setup.py


References

[Wei et al. (2022)](https://arxiv.org/pdf/2201.11903). Chain-of-Thought Prompting in LLMs.
 
Wang et al. (2022). Self-Consistency in Chain-of-Thought Reasoning.
 
Zelikman et al. (2022). STaR: Self-Taught Reasoner.
 
Schulman et al. (2017). Proximal Policy Optimization.
 
DeepSeek R1 & TinyZero: GitHub:
Jiayi-Pan/TinyZero.
 
Shao et al., 2024: DeepSeekMath introducing GRPO.
 
Sasha Rush’s Repository 

GSM8K
