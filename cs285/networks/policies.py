import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            # 离散动作空间
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            # 连续动作空间
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # DONE: implement get_action
        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs)
        # Using rsample to allow gradients to pass through the sample

        if self.discrete:
            action = action_distribution.sample()
        else:
            action = action_distribution.rsample()
        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # DONE: define the forward pass for a policy with a discrete action space.
            action_distribution = distributions.Categorical(
                logits=self.logits_net(obs)
            )
        else:
            # DONE: define the forward pass for a policy with a continuous action space.
            action_distribution = distributions.Normal(
                loc=self.mean_net(obs),
                scale=torch.exp(self.logstd)
            )
        return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # DONE: implement the policy gradient actor update.

        # 1. 前向传播得到策略分布
        action_distribution = self.forward(obs)

        # 2. 计算选择动作的对数概率
        if self.discrete:
            log_prob = action_distribution.log_prob(actions)
        else:
            # For continuous action spaces, actions are typically multidimensional,
            # so we sum the log prob across dimensions
            log_prob = action_distribution.log_prob(actions).sum(dim=-1)
            # 为什么要sum？
            # 连续动作通常是多维的（如机器人的多个关节）
            # 假设各维度独立：P(a₁,a₂) = P(a₁) × P(a₂)
            # 对数空间：log P(a₁,a₂) = log P(a₁) + log P(a₂)
        
        # 3. 计算策略梯度 gradient = E[log π(a|s) × A(s,a)]
        # 4. 梯度上升（通过最小化负梯度）loss = -gradient
        loss = -(log_prob * advantages).mean()  # Negative for gradient ascent

        # Backward pass on the loss
        # 清空梯度：防止梯度累积
        self.optimizer.zero_grad()  
        # 反向传播：计算损失对参数的梯度
        loss.backward()         # 计算 ∇θ(-policy_gradient)
        # 参数更新：使用优化器（Adam）更新网络参数
        self.optimizer.step()   # θ ← θ - lr × ∇θ(-policy_gradient)
                                # θ ← θ + lr × ∇θ(policy_gradient)  ← 这就是梯度上升！

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
