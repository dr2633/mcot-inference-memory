import numpy as np


def reward_correctness(completions, reference_answers):
    """
    Rewards completions based on correctness.

    Args:
        completions (List[str]): Generated model outputs.
        reference_answers (List[str]): Ground-truth answers.

    Returns:
        List[float]: Rewards for each completion.
    """
    rewards = []
    for completion, gold_answer in zip(completions, reference_answers):
        reward = 1.0 if completion.strip() == gold_answer.strip() else -1.0
        rewards.append(reward)
    return rewards


def reward_efficiency(completions, reference_answers):
    """
    Rewards efficiency by penalizing excessive token usage.

    Args:
        completions (List[str]): Generated model outputs.
        reference_answers (List[str]): Ground-truth answers.

    Returns:
        List[float]: Efficiency-based rewards.
    """
    rewards = []
    for completion, gold_answer in zip(completions, reference_answers):
        gold_length = len(gold_answer.split())
        completion_length = len(completion.split())

        # Penalize unnecessary verbosity (excess token usage)
        efficiency_penalty = max(0, (completion_length - gold_length) / gold_length)
        reward = 1.0 - efficiency_penalty
        rewards.append(reward)

    return rewards


def reward_formatting(completions, retrieved_memories):
    """
    Rewards formatting consistency by checking alignment with retrieved memory.

    Args:
        completions (List[str]): Generated model outputs.
        retrieved_memories (List[str]): Previously stored CoT responses.

    Returns:
        List[float]: Formatting rewards.
    """
    rewards = []
    for idx, completion in enumerate(completions):
        memory_text = retrieved_memories[idx] if idx < len(retrieved_memories) else ""
        similarity = text_similarity(completion, memory_text) if memory_text else 0.0

        # Higher similarity to past responses gets a better reward
        reward = similarity
        rewards.append(reward)

    return rewards


def text_similarity(text1, text2):
    """
    Simple similarity metric: Jaccard similarity over word tokens.

    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        float: Similarity score between 0 and 1.
    """
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def combined_reward(completions, reference_answers, retrieved_memories, weights):
    """
    Computes a combined reward function based on correctness, efficiency, and formatting.

    Args:
        completions (List[str]): Generated model outputs.
        reference_answers (List[str]): Ground-truth answers.
        retrieved_memories (List[str]): Retrieved memory-based responses.
        weights (dict): Weights for different reward components.

    Returns:
        List[float]: Combined rewards.
    """
    correctness_scores = reward_correctness(completions, reference_answers)
    efficiency_scores = reward_efficiency(completions, reference_answers)
    formatting_scores = reward_formatting(completions, retrieved_memories)

    combined_rewards = []
    for i in range(len(completions)):
        reward = (
                weights["correctness"] * correctness_scores[i] +
                weights["efficiency"] * efficiency_scores[i] +
                weights["formatting"] * formatting_scores[i]
        )
        combined_rewards.append(np.clip(reward, -1.0, 1.0))  # Clip to [-1, 1]

    return combined_rewards
