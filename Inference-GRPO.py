import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any

class InferenceGRPOAgent:
    """
    Demonstrates how to do test-time preference optimization with memory (IPO).
    This example uses random 'rewards' for simplicity.
    """
    def __init__(self, policy_model, old_policy_model, memory_store, kl_coeff=0.1, clip_range=0.2):
        """
        Args:
            policy_model (torch.nn.Module): Current policy model.
            old_policy_model (torch.nn.Module): A frozen/slow-updating reference model.
            memory_store (MemoryStore): For retrieving user-specific memory to shape the policy.
            kl_coeff (float): Coefficient for KL divergence penalty.
            clip_range (float): PPO/GRPO clip range for ratio.
        """
        self.policy_model = policy_model
        self.old_policy_model = old_policy_model
        self.memory_store = memory_store
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range

    def retrieve_memory_for_user(self, user_id: str):
        """
        In practice, you might pass a query or session ID to retrieve relevant memory.
        """
        # For demonstration, let's just do a naive retrieval
        results = self.memory_store.retrieve_relevant_memory(user_id)
        return results

    def forward_inference(self, inputs: Dict[str, torch.Tensor], user_id: str) -> Dict[str, torch.Tensor]:
        """
        1) Retrieve user memory
        2) Forward pass with policy_model
        3) Compare with old_policy_model for ratio
        4) Possibly update policy_model at inference with a quick gradient step
        """
        # 1. Retrieve memory
        user_memory = self.retrieve_memory_for_user(user_id)
        # In a real setting, you'd incorporate user_memory into the model input, e.g. as embeddings or cross-attention context

        # 2. Forward pass
        current_logits = self.policy_model(inputs["obs"])  # e.g. shape [batch, action_dim]
        old_logits = self.old_policy_model(inputs["obs"]).detach()

        # Compute distributions
        current_dist = F.softmax(current_logits, dim=-1)
        old_dist = F.softmax(old_logits, dim=-1)

        # 3. Compute ratio for a GRPO-like update
        ratio = (current_dist / (old_dist + 1e-8)).clamp(0.0, 10.0)  # avoid dividing by zero
        # Mock advantage or reward
        advantage = torch.rand_like(ratio)  # In practice, you'd compute real advantage or preference

        # 4. Clipped objective
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
        policy_loss = -torch.min(unclipped, clipped).mean()

        # Add KL penalty
        kl_div = F.kl_div(old_dist.log(), current_dist, reduction='batchmean')
        total_loss = policy_loss + self.kl_coeff * kl_div

        # 5. Inference-time update (tiny gradient step, if allowed)
        # NOTE: This is unorthodox—most RL setups do not do gradient updates "live" at inference.
        # This demonstrates how "IPO" might do so.
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "action_probs": current_dist.detach(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
        }


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    class SimplePolicyModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=5):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    # Instantiate memory store, policy models
    memory_store = MemoryStore(json_path="user_memory.json")
    policy_model = SimplePolicyModel()
    old_policy_model = SimplePolicyModel()
    old_policy_model.load_state_dict(policy_model.state_dict())  # initialize old model with same weights

    agent = InferenceGRPOAgent(policy_model, old_policy_model, memory_store)

    # Dummy input
    batch_obs = torch.randn((2, 10))  # e.g. batch of size 2, input_dim=10
    inputs = {"obs": batch_obs}

    # "Inference" call
    user_id = "user123"  # or any relevant ID for retrieving memory
    output = agent.forward_inference(inputs, user_id)
    print("Action probabilities:", output["action_probs"])
    print("Policy loss:", output["policy_loss"])
    print("KL div:", output["kl_div"])
