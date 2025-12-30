import math
import torch
import torch.nn as nn


class LinearLoRA(nn.Module):
    """
    适用于全连接层的LoRA模块。
    设计为"权重低秩增量"形式，便于直接与基础权重相加，
    同时在前向中记录down/up激活以供梯度投影使用。
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = None):
        super().__init__()
        assert rank > 0
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()

        # 缓存前向激活，供特征矩阵统计
        self.down_ = None  # 输入激活
        self.up_ = None    # 经过A后的激活

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def delta_weight(self):
        # 返回可直接与基础权重相加的低秩增量矩阵
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def cache_activations(self, x: torch.Tensor):
        # 记录down/up激活（不参与梯度）
        self.down_ = x.detach()
        self.up_ = self.lora_A(x.detach())

    def forward(self, base_layer: nn.Linear, x: torch.Tensor):
        # 经典LoRA前向：基础输出 + 低秩增量
        self.cache_activations(x)
        return base_layer(x) + self.lora_B(self.up_) * self.scaling

