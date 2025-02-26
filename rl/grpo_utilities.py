import os
import torch
import json
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------
# Reward Aggregation Utility
# ----------------------------------------------

def aggregate_rewards(completions, reference_answers, retrieved_memories, reward_weights, reward_functions):
    """
    Computes the final reward value by aggregating correctness, efficiency, and formatting.

    Args:
        completions (List[str]): Model-generated outputs.
        reference_answers (List[str]): Ground-truth answers.
        retrieved_memories (List[str]): Memory-augmented responses.
        reward_weights (dict): Weights for different reward types.
        reward_functions (dict): Dictionary of reward functions.

    Returns:
        List[float]: Aggregated reward values for each completion.
    """
    rewards = []
    for idx, completion in enumerate(completions):
        reward = 0.0
        for key, reward_fn in reward_functions.items():
            if key in reward_weights:
                reward += reward_weights[key] * \
                          reward_fn([completion], [reference_answers[idx]], [retrieved_memories[idx]])[0]

        # Clip to avoid extreme values
        reward = np.clip(reward, -1.0, 1.0)
        rewards.append(reward)

    return rewards


# ----------------------------------------------
# Logging and Tracking
# ----------------------------------------------
def log_training_progress(step, rewards, loss, log_file="training_log.json"):
    """
    Logs training progress, including step count, reward distribution, and loss.

    Args:
        step (int): Current training step.
        rewards (List[float]): Reward values from the latest batch.
        loss (float): Model loss value.
        log_file (str): Path to the log file.

    Returns:
        None
    """
    log_data = {
        "step": step,
        "average_reward": np.mean(rewards),
        "reward_variance": np.var(rewards),
        "loss": loss,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logging.info(f"Step {step}: Avg Reward: {log_data['average_reward']:.3f}, Loss: {loss:.4f}")

    # Append to JSON log
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_data)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


# ----------------------------------------------
# Checkpointing Utilities
# ----------------------------------------------
def save_checkpoint(model, tokenizer, optimizer, step, output_dir="grpo_checkpoints"):
    """
    Saves a training checkpoint for resuming later.

    Args:
        model (transformers.PreTrainedModel): Model instance.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        step (int): Current training step.
        output_dir (str): Directory to save checkpoint.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")

    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    logging.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, tokenizer, optimizer, checkpoint_path):
    """
    Loads a training checkpoint.

    Args:
        model (transformers.PreTrainedModel): Model instance.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        checkpoint_path (str): Path to the checkpoint directory.

    Returns:
        int: Resumed training step.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    model.load_pretrained(checkpoint_path)
    tokenizer.load_pretrained(checkpoint_path)
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))

    step = int(checkpoint_path.split("-")[-1])
    logging.info(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}")

    return step
