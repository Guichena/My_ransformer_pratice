<<<<<<< HEAD
# My Own Transformers

从零实现的 Transformer 架构，用于英译中机器翻译任务。本项目基于论文 *Attention Is All You Need* 的原始规范，包含完整的编码器-解码器架构、多头注意力机制、位置编码、残差连接和学习率预热调度。

这是一个教育性质的参考实现，适合用于学习 Transformer 内部机制、进行实验或作为研究起点。

<div align="center">
  <img src="model architecture.png" width="55%">
</div>

---

## 目录

- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [新增功能](#新增功能)
- [快速开始](#快速开始)
- [训练模式](#训练模式)
- [配置参数](#配置参数)
- [数据准备](#数据准备)
- [实现细节](#实现细节)
- [工具脚本](#工具脚本)
- [参考资料](#参考资料)

---

## 核心特性

### 模型架构
- ✅ **标准 Transformer 编码器-解码器**：6 层 × 8 头（可配置）
- ✅ **多头自注意力机制**：缩放点积注意力，支持并行计算
- ✅ **位置编码**：正弦/余弦位置嵌入（最大长度 5000）
- ✅ **前馈网络**：两层全连接 + ReLU 激活
- ✅ **残差连接 + 层归一化**：每个子层后应用
- ✅ **Dropout 正则化**：防止过拟合

### 训练系统
- ✅ **自定义优化器**：Adam + 预热 + 逆平方根衰减学习率调度
- ✅ **Teacher Forcing**：训练时使用真实目标序列
- ✅ **自回归生成**：推理时逐 token 生成翻译
- ✅ **BLEU 评估**：字符级 BLEU 分数（语言无关）
- ✅ **Checkpoint 管理**：保存模型、优化器状态和指标
- ✅ **早停机制**：验证集 BLEU 不提升时自动停止
- ✅ **TensorBoard 可视化**：实时监控损失和 BLEU 曲线

### 分词系统（双模式）
- ✅ **字符级分词器**：小词表（~2700），适合小数据集（<10K 样本）
- ✅ **TikToken 分词器**：BPE 子词分词（100K 词表），适合大数据集

---

## 项目架构

```
My_Own_Transformers/
│
├── src/                          # 核心源代码
│   ├── main.py                   # 训练入口 + Transformer 类
│   ├── models.py                 # 编码器、解码器、注意力层
│   ├── data.py                   # 数据集、数据加载器
│   ├── utils.py                  # 工具函数（mask、BLEU、分词）
│   ├── optimizer.py              # 自定义优化器（带预热）
│   ├── config.py                 # 配置管理
│   └── args.py                   # 命令行参数定义
│
├── data/                         # 数据文件
│   ├── train_clean.tsv           # 训练集（18,746 条）
│   ├── val_clean.tsv             # 验证集（2,343 条）
│   ├── test_clean.tsv            # 测试集（2,344 条）
│   ├── char_tokenizer.json       # 字符级词表（2,694 词汇）
│   ├── demo_train.tsv            # Demo 训练集（10 条）
│   ├── demo_val.tsv              # Demo 验证集（3 条）
│   └── demo_test.tsv             # Demo 测试集（3 条）
│
├── checkpoints/                  # 模型检查点
├── runs/                         # TensorBoard 日志
│
├── char_tokenizer.py             # 字符级分词器实现
├── split_dataset.py              # 数据集划分工具
├── test_demo_translation.py      # Demo 模型测试脚本
├── test_translation_full.py      # 完整模型测试脚本
├── train_full_dataset.bat        # Windows 训练脚本
│
├── requirements.txt              # Python 依赖
└── README.md                     # 项目文档
```

**依赖**：`torch`, `tiktoken`, `tqdm`, `tensorboard`

---

## 新增功能

### 1. 快速评估模式 ⚡
**问题**：完整评估 2343 条验证数据需要 5-10 分钟，训练速度慢。

**解决方案**：添加 `--quick_eval` 参数
- 只评估前 3 个 batch（约 96 条数据）
- 展示 5 个翻译样例（原文 + 标准翻译 + 模型翻译）
- 评估时间从 5-10 分钟降至 10-20 秒
- BLEU 分数基于 96 条数据（足够看趋势）

```bash
# 启用快速评估
python src/main.py --quick_eval ...
```

### 2. 字符级分词器
**问题**：TikToken（100K 词表）需要大量数据才能训练好。

**解决方案**：实现字符级分词器
- 词表大小：~2700（英文 + 中文字符）
- 特殊 token：PAD(0), BOS(1), EOS(2), UNK(3)
- 无 OOV 问题
- 模型大小：~10MB（vs TikToken 的 479MB）
- 适合小数据集（<10K 样本）

```python
# char_tokenizer.py
class CharTokenizer:
    def encode(self, text):
        # 字符级编码
    def decode(self, tokens):
        # 字符级解码
```

### 3. 翻译样例展示
**问题**：训练时只能看到 BLEU 分数，无法直观看到翻译效果。

**解决方案**：评估时自动展示翻译样例
```
[Sample 0]
  原文:     That red dress suited her.
  标准翻译:  那件红色的洋装适合她。
  生成翻译: 那件红色的洋装适合她。
```

### 4. 模型配置自动推断
**问题**：测试时需要手动指定模型配置参数。

**解决方案**：从 checkpoint 自动推断配置
```python
# 从权重 shape 推断
inferred_d_model = state_dict['encoder.embedding.weight'].shape[1]
inferred_num_layers = len([k for k in state_dict.keys()
                           if 'encoder.layers' in k])
```

### 5. 交互式测试脚本
**功能**：
- 自动加载最新 checkpoint
- 测试预设句子
- 交互式翻译模式（输入英文 → 输出中文）

```bash
python test_translation_full.py
```

---

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### Demo 训练（10 个样本）
```bash
python src/main.py --use_demo_data --device gpu
```

### 完整数据集训练（18K 样本）
```bash
python src/main.py \
    --train_file data/train_clean.tsv \
    --val_file data/val_clean.tsv \
    --tokenizer_type char \
    --char_tokenizer_path data/char_tokenizer.json \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --d_ff 1024 \
    --batch_size 32 \
    --num_epochs 50 \
    --quick_eval \
    --tensorboard \
    --device cuda
```

### 启动 TensorBoard
```bash
tensorboard --logdir=runs
# 访问 http://localhost:6006
```

### 测试翻译
```bash
python test_translation_full.py
```

---

## 训练模式

### 1. Demo 模式（快速验证）
```bash
python src/main.py --use_demo_data --device cpu
```
- 数据：10 训练 + 3 验证 + 3 测试
- 用途：验证代码正确性
- 预期 BLEU：~0.27（记忆训练数据）

### 2. 快速评估模式（推荐）
```bash
python src/main.py --quick_eval ...
```
- 每个 epoch 评估时间：10-20 秒
- 展示 5 个翻译样例
- BLEU 基于 96 条数据

### 3. 完整评估模式
```bash
python src/main.py ...  # 不加 --quick_eval
```
- 评估全部验证集（2343 条）
- 每个 epoch 评估时间：5-10 分钟
- 准确的 BLEU 分数

### 4. 恢复训练
```bash
python src/main.py --resume checkpoints/checkpoint_epoch_10.pt
```

---

## 配置参数

### 模型参数
| 参数 | 说明 | 默认值 | 推荐值（小数据集） |
|------|------|--------|-------------------|
| `--d_model` | 嵌入维度 | 512 | 256 |
| `--num_layers` | 编码器/解码器层数 | 6 | 4 |
| `--num_heads` | 注意力头数 | 8 | 8 |
| `--d_ff` | 前馈网络维度 | 2048 | 1024 |
| `--dropout` | Dropout 比率 | 0.1 | 0.1 |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 批次大小 | 32 |
| `--num_epochs` | 训练轮数 | 10 |
| `--learning_rate` | 学习率缩放因子 | 1.0 |
| `--warmup_steps` | 预热步数 | 60 |
| `--patience` | 早停耐心值 | 5 |
| `--device` | 设备（cuda/cpu） | cuda |

### 评估参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--quick_eval` | 快速评估模式 | False |
| `--skip_train_eval` | 跳过训练集评估 | False |
| `--max_eval_batches` | 最大评估 batch 数 | None |
| `--eval_interval` | 评估间隔（epoch） | 1 |

### 数据参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_file` | 训练数据路径 | None |
| `--val_file` | 验证数据路径 | None |
| `--tokenizer_type` | 分词器类型（char/tiktoken） | char |
| `--char_tokenizer_path` | 字符分词器路径 | data/char_tokenizer.json |
| `--max_seq_len` | 最大序列长度 | 512 |
| `--use_demo_data` | 使用 demo 数据 | False |

### 可视化参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tensorboard` | 启用 TensorBoard | False |
| `--tensorboard_dir` | TensorBoard 日志目录 | runs |
| `--log_interval` | 日志打印间隔 | 100 |

---

## 数据准备

### 数据格式
UTF-8 编码的 TSV 文件，每行格式：`<源语言>\t<目标语言>`

```tsv
That red dress suited her.	那件红色的洋装适合她。
We ran down the hill.	我们跑下山。
Knowledge is power.	知识就是力量。
```

### 数据处理流程
1. **加载**：`load_data_from_file()` 读取 TSV
2. **分词**：`encode_text()` 添加 BOS/EOS，截断到 max_seq_len
3. **批处理**：`pad_sequence()` 填充到相同长度
4. **Teacher Forcing**：生成 `(tgt_input, tgt_output)` 对

### 创建字符级分词器
```bash
python char_tokenizer.py
```

### 划分数据集
```bash
python split_dataset.py
# 输出：train_clean.tsv (80%), val_clean.tsv (10%), test_clean.tsv (10%)
```

---

## 实现细节

### Transformer 模型结构

完整的编码器-解码器架构，遵循原始论文设计：

```
输入序列 (英文)
    ↓
[编码器]
    ├─ 词嵌入 (Embedding)
    ├─ 位置编码 (Positional Encoding)
    └─ N × 编码器层
        ├─ 多头自注意力 (Multi-Head Self-Attention)
        ├─ 残差连接 + 层归一化
        ├─ 前馈网络 (Feed-Forward)
        └─ 残差连接 + 层归一化
    ↓
编码器输出 (上下文表示)
    ↓
[解码器]
    ├─ 词嵌入 (Embedding)
    ├─ 位置编码 (Positional Encoding)
    └─ N × 解码器层
        ├─ 掩码多头自注意力 (Masked Multi-Head Self-Attention)
        ├─ 残差连接 + 层归一化
        ├─ 编码器-解码器注意力 (Cross-Attention)
        ├─ 残差连接 + 层归一化
        ├─ 前馈网络 (Feed-Forward)
        └─ 残差连接 + 层归一化
    ↓
线性层 + Softmax
    ↓
输出序列 (中文)
```

---

### 1. 编码器实现 (Encoder)

#### 结构
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)

        # N 个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)
```

#### 前向传播
```python
def forward(self, src, src_mask=None):
    # 1. 词嵌入 + 缩放
    x = self.embedding(src) * math.sqrt(self.d_model)

    # 2. 添加位置编码
    x = self.pos_encoding(x)

    # 3. 通过 N 个编码器层
    for layer in self.layers:
        x = layer(x, src_mask)

    # 4. 最终层归一化
    return self.norm(x)
```

#### 关键点
- **词嵌入缩放**：乘以 √d_model，使位置编码和词嵌入的量级相当
- **位置编码**：固定的正弦/余弦函数，不需要训练
- **堆叠层**：6 层（默认），每层独立参数

---

### 2. 解码器实现 (Decoder)

#### 结构
```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)

        # N 个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)

        # 输出投影层
        self.linear = nn.Linear(d_model, vocab_size)
```

#### 前向传播
```python
def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
    # 1. 词嵌入 + 缩放
    x = self.embedding(tgt) * math.sqrt(self.d_model)

    # 2. 添加位置编码
    x = self.pos_encoding(x)

    # 3. 通过 N 个解码器层
    for layer in self.layers:
        x = layer(x, enc_output, src_mask, tgt_mask)

    # 4. 最终层归一化
    x = self.norm(x)

    # 5. 投影到词表
    return self.linear(x)
```

#### 关键点
- **三种注意力**：自注意力、交叉注意力、前馈网络
- **掩码机制**：防止看到未来的 token
- **输出投影**：将 d_model 维度映射到词表大小

---

### 3. 注意力机制

#### 3.1 多头注意力 (Multi-Head Attention)

**核心思想**：将注意力分成多个"头"，每个头学习不同的表示子空间。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影矩阵
        self.w_o = nn.Linear(d_model, d_model)
```

**计算流程**：
```python
def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)

    # 1. 线性投影并分成多头
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    # 2. 缩放点积注意力
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    # 3. 应用 mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 5. 加权求和
    attn_output = torch.matmul(attn_weights, V)

    # 6. 合并多头
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, -1, self.d_model)

    # 7. 输出投影
    return self.w_o(attn_output)
```

**数学公式**：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
其中 head_i = Attention(Q×W^Q_i, K×W^K_i, V×W^V_i)
```

#### 3.2 缩放点积注意力 (Scaled Dot-Product Attention)

**为什么要缩放**？
- 点积结果的方差与 d_k 成正比
- 当 d_k 很大时，点积值会很大，导致 softmax 梯度很小
- 除以 √d_k 可以稳定梯度

**示例**（d_k=64）：
```
未缩放：scores = [100, 80, 60] → softmax → [1.0, 0.0, 0.0]（梯度消失）
缩放后：scores = [12.5, 10, 7.5] → softmax → [0.7, 0.2, 0.1]（梯度正常）
```

#### 3.3 位置编码 (Positional Encoding)

**为什么需要**？
- 注意力机制本身是置换不变的（permutation invariant）
- 需要显式地注入位置信息

**实现**：
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 计算分母项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # 偶数维度使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加 batch 维度
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

**数学公式**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**优点**：
- 固定函数，不需要训练
- 可以外推到更长的序列
- 相对位置关系可以通过三角函数性质表示

---

### 4. 编码器层 (EncoderLayer)

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # 两个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层 1：多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接

        # 子层 2：前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))    # 残差连接

        return x
```

**关键设计**：
- **残差连接**：`x = norm(x + sublayer(x))`，缓解梯度消失
- **层归一化**：在残差连接后应用，稳定训练
- **Dropout**：在子层输出后、残差连接前应用

---

### 5. 解码器层 (DecoderLayer)

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        # 掩码多头自注意力
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)

        # 编码器-解码器注意力
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 子层 1：掩码多头自注意力
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层 2：编码器-解码器注意力
        attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 子层 3：前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
```

**三种注意力**：
1. **掩码自注意力**：解码器关注已生成的 token（带 look-ahead mask）
2. **交叉注意力**：解码器关注编码器输出（Q 来自解码器，K/V 来自编码器）
3. **前馈网络**：位置独立的非线性变换

### 训练系统（main.py）

#### 1. 训练步骤
```python
def train_step(model, src, tgt_input, tgt_output, ...):
    # 1. 创建 mask
    # 2. 前向传播
    # 3. 计算损失（交叉熵）
    # 4. 反向传播
    # 5. 更新参数
```

#### 2. 评估
```python
def evaluate(model, tokenizer, eval_data, quick_eval=False):
    # 1. 自回归生成翻译
    # 2. 计算 BLEU 分数
    # 3. 展示翻译样例
```

#### 3. 自回归生成
```python
def translate(model, src, tokenizer):
    # 从 BOS 开始
    # 循环：
    #   1. 前向传播
    #   2. 预测下一个 token
    #   3. 拼接到序列
    #   4. 如果是 EOS，停止
```

### 优化器（optimizer.py）
```python
class TransformerOptimizer:
    # 学习率调度：
    # lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

### Mask 机制（utils.py）
```python
# Padding mask：屏蔽 PAD token
# Look-ahead mask：防止看到未来 token
# 组合 mask：用于解码器
```

---

## 工具脚本

### 1. char_tokenizer.py
字符级分词器实现
```bash
python char_tokenizer.py
```

### 2. split_dataset.py
数据集划分（80/10/10）
```bash
python split_dataset.py
```

### 3. test_demo_translation.py
测试 demo 模型（<10MB）
```bash
python test_demo_translation.py
```

### 4. test_translation_full.py
测试完整模型（>10MB）
```bash
python test_translation_full.py
```

### 5. train_full_dataset.bat
Windows 一键训练脚本
```bash
train_full_dataset.bat
```

---

## 实验结果与建议

### 数据量 vs 性能
| 数据量 | BLEU | 说明 |
|--------|------|------|
| 10 样本 | 0.27 | 记忆训练数据，无泛化能力 |
| 150 样本 | 0.00 | 数据不足，无法学习 |
| 1K-10K 样本 | 0.05-0.15 | 开始学习模式 |
| 10K-100K 样本 | 0.15-0.30 | 较好的翻译质量 |
| >100K 样本 | >0.30 | 高质量翻译 |

### 模型配置建议
| 数据量 | d_model | num_layers | d_ff | 说明 |
|--------|---------|------------|------|------|
| <1K | 64 | 1-2 | 128 | 小模型，防止过拟合 |
| 1K-10K | 128-256 | 2-4 | 512-1024 | 中等模型 |
| 10K-100K | 256-512 | 4-6 | 1024-2048 | 标准模型 |
| >100K | 512 | 6 | 2048 | 完整模型 |

### 关键发现
1. **Transformer 是数据饥渴型模型**：需要大量数据才能学好
2. **字符级分词适合小数据集**：词表小，模型容量小
3. **快速评估加速训练**：96 条数据足够看趋势
4. **早停防止过拟合**：小数据集容易过拟合

---

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Harvard NLP Transformer 注解：<https://nlp.seas.harvard.edu/2018/04/03/attention.html>
- TikToken 分词器：<https://github.com/openai/tiktoken>

---

=======
# My_Own_Transformers
My own and noob implementation of Transformer Model Family.
>>>>>>> 85c67de (Initial commit)
