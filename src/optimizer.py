import torch
import torch.optim as optim
import math
from typing import Optional


class TransformerOptimizer:
    def __init__(self, model, model_size, factor=1.0, warmup=4000, optimizer=None):
        """
        初始化 Transformer 优化器。
        参数:
        - model: 模型参数
        - model_size: 模型维度
        - factor: 学习率缩放因子
        - warmup: warmup 步数
        - optimizer: 底层优化器
        """
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0

        if optimizer is None:
            optimizer = optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            )

        self.inner_optimizer = optimizer

    def step(self, closure=None):
        """
        执行单步优化。
        参数:
        - closure: 重新评估模型并返回 loss 的闭包
        返回:
        - 损失值
        """
        lr = self.rate()
        for param_group in self.inner_optimizer.param_groups:
            param_group["lr"] = lr

        self._step += 1
        return self.inner_optimizer.step(closure)

    def rate(self):
        """
        计算当前学习率。
        返回:
        - 学习率
        """
        # 确保 step_num 非 0
        step_num = max(1, self._step)
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step_num ** (-0.5), step_num * self.warmup ** (-1.5))
        )

    def zero_grad(self, set_to_none=False):
        """
        清空所有参数的梯度。
        参数:
        - set_to_none: 若为 True，将梯度设为 None（而非 0）
        """
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        返回优化器的状态字典。
        返回:
        - 状态字典
        """
        return {
            "step_num": self._step,
            "model_size": self.model_size,
            "factor": self.factor,
            "warmup": self.warmup,
            "optimizer": self.inner_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        加载优化器状态。
        参数:
        - state_dict: 状态字典
        """
        self._step = state_dict["step_num"]
        self.model_size = state_dict["model_size"]
        self.factor = state_dict["factor"]
        self.warmup = state_dict["warmup"]
        self.inner_optimizer.load_state_dict(state_dict["optimizer"])


def get_optimizer(model, model_size, factor=2.0, warmup=4000):
    """
    创建 Transformer 优化器。
    参数:
    - model: 模型
    - model_size: 模型维度
    - factor: 学习率缩放因子
    - warmup: warmup 步数
    返回:
    - TransformerOptimizer 实例
    """
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return TransformerOptimizer(
        model=model,
        model_size=model_size,
        factor=factor,
        warmup=warmup,
        optimizer=optimizer,
    )
