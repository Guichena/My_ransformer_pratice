import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Encoding 模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化 Positional Encoding 模块。
        参数:
        - d_model: 特征维度（embedding 维度）
        - max_len: 最大序列长度
        """
        super().__init__()

        # 初始化为 0 的位置编码矩阵，形状 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，形状变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为持久 buffer，不需要梯度
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向计算。
        参数:
        - x: 输入张量，形状 [batch_size, seq_len, d_model]
        返回:
        - 加上位置编码后的张量，形状与输入相同
        """
        # 获取输入序列长度
        seq_len = x.size(1)

        # 取与序列长度匹配的位置编码
        position_encoding = self.pe[:, :seq_len]

        # 通过广播将位置编码加到输入上
        return x + position_encoding


# Multi-Head Attention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化 Multi-Head Attention 模块。
        参数:
        - d_model: 输入特征维度
        - num_heads: 注意力头数量
        - dropout: Dropout 概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass.
        Parameters:
        - q: Query tensor with shape [batch_size, seq_len, d_model]
        - k: Key tensor with shape [batch_size, seq_len, d_model]
        - v: Value tensor with shape [batch_size, seq_len, d_model]
        - mask: Mask tensor with shape [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        device = q.device

        # 线性映射并拆分 heads，得到 [batch_size, num_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用 mask
        if mask is not None:
            # 确保 mask 位于正确 device 上
            mask = mask.to(device)

            # 处理 mask 维度，确保其为 4 维
            if mask.dim() > 4:
                mask = mask.squeeze(2)

            # 将 mask 扩展到所有 heads
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.num_heads, -1, -1)

            # 将 mask 应用于 scores（被 mask 的位置在 softmax 前置为 -inf）
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重 [batch_size, num_heads, q_len, k_len]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 计算输出 [batch_size, num_heads, q_len, d_k]
        output = torch.matmul(attn, v)

        # 维度变换并合并 heads，得到 [batch_size, q_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output


# Position-wise Feed-Forward 网络模块
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化 Position-wise Feed-Forward 网络。
        参数:
        - d_model: 输入特征维度
        - d_ff: FFN 隐藏层维度
        - dropout: Dropout 概率
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        前向计算。
        参数:
        - x: 输入张量，形状 [batch_size, seq_len, d_model]
        返回:
        - 输出张量，形状 [batch_size, seq_len, d_model]
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# Encoder 层模块
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化 Encoder 层。
        参数:
        - d_model: 输入特征维度
        - num_heads: 注意力头数量
        - d_ff: FFN 隐藏层维度
        - dropout: Dropout 概率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向计算。
        参数:
        - x: 输入张量，形状 [batch_size, seq_len, d_model]
        - mask: mask 张量
        返回:
        - 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 自注意力层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# Encoder 模块
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.1,
        max_len=5000,
        device="cpu",
    ):
        """
        Initialize encoder module.
        Parameters:
        - vocab_size: Vocabulary size
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - num_layers: Number of encoder layers
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        - max_len: Maximum sequence length
        - device: Device (e.g., "cpu" or "cuda")
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向计算。
        参数:
        - x: 输入张量，形状 [batch_size, seq_len]
        - mask: mask 张量
        返回:
        - 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 词向量与位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# Decoder 层模块
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化 Decoder 层。
        参数:
        - d_model: 输入特征维度
        - num_heads: 注意力头数量
        - d_ff: FFN 隐藏层维度
        - dropout: Dropout 概率
        """
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向计算。
        参数:
        - x: Decoder 输入张量，形状 [batch_size, seq_len, d_model]
        - enc_output: Encoder 输出张量，形状 [batch_size, seq_len, d_model]
        - src_mask: 源序列 mask
        - tgt_mask: 目标序列 mask
        返回:
        - 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # Masked 自注意力层
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Encoder-Decoder 注意力层
        # src_mask 形状: [batch_size, 1, 1, src_len]
        # 该形状支持在 heads 与 query 维度上的广播
        attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # 前馈网络层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


# Decoder 模块
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.1,
        max_len=5000,
        device="cpu",
    ):
        """
        初始化 Decoder 模块。
        参数:
        - vocab_size: 词表大小
        - d_model: 输入特征维度
        - num_heads: 注意力头数量
        - num_layers: Decoder 层数
        - d_ff: FFN 隐藏层维度
        - dropout: Dropout 概率
        - max_len: 最大序列长度
        - device: 设备（如 "cpu" 或 "cuda"）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向计算。
        参数:
        - x: Decoder 输入张量，形状 [batch_size, seq_len]
        - enc_output: Encoder 输出张量，形状 [batch_size, seq_len, d_model]
        - src_mask: 源序列 mask
        - tgt_mask: 目标序列 mask
        返回:
        - 输出张量，形状 [batch_size, seq_len, vocab_size]
        """
        # 词向量与位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过 N 层 Decoder
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        # 最后一层 LayerNorm
        x = self.norm(x)

        # 线性映射到词表大小
        return self.linear(x)
