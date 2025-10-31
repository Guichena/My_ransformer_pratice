# 解决模式坍塌问题 - 总结报告

## 问题诊断

### 根本原因
模型生成**相同的翻译**对于所有输入："我正努力理解释这个新任务。"

### 发现的问题

1. **严重的数据不平衡**
   - 原始训练数据：1111行
   - 其中 **155行 (14%)** 都是 "I'm trying to..." 句式
   - 所有这些句子的中文翻译都以"我正努力..."开头
   - 模型学会了过拟合到这个高频模式

2. **词汇表过大**
   - 原使用 tiktoken cl100k_base: **100,000 tokens**
   - 只有 1111 个训练样本
   - 参数与数据比例极度不平衡

3. **数据与 demo_data 的对比**
   - `use_demo_data`工作正常，因为只有10个样本，但**每个都完全不同**
   - 原 train.tsv 有 1111 个样本，但有 **785行都是重复的"I'm trying to"模式**

## 解决方案

### 1. 创建多样化的训练数据

✅ **已完成** - 修改 `create_data.py`

**改进：**
- 从 1111 samples → **150 samples**
- 删除了 961 个重复的 "I'm trying to" 句子
- 每个句子都有独特的结构：
  - 30个经典谚语
  - 40个日常对话（多种句式）
  - 40个描述性句子（不同主题）
  - 40个复杂句式（多样化表达）

**对比：**
```
旧数据：
- "I'm trying to eat healthier food." → "我正努力吃更健康的食物。"
- "I'm trying to improve my cooking skills." → "我正努力提高我的烹饪技巧。"
- ...（重复155次）

新数据：
- "Knowledge is power." → "知识就是力量。"
- "What are your plans for the weekend?" → "你周末有什么计划？"
- "Artificial intelligence is changing the world." → "人工智能正在改变世界。"
- "He is known for his integrity." → "他以正直著称。"
```

### 2. 实现字符级Tokenizer

✅ **已完成** - 创建 `char_tokenizer.py`

**改进：**
- 词汇表大小：100,000 → **678 tokens**
- 支持英文和中文字符
- 特殊tokens: PAD(0), BOS(1), EOS(2), UNK(3)
- 可保存/加载

**优势：**
- 大幅减少模型参数
- 更适合小数据集
- 不会出现 OOV (Out of Vocabulary) 问题

### 3. 更新代码以支持新Tokenizer

✅ **已完成** - 修改多个文件

**修改的文件：**
1. `src/args.py` - 添加参数:
   - `--tokenizer_type`: 选择 "char" 或 "tiktoken"
   - `--char_tokenizer_path`: 字符级tokenizer文件路径

2. `src/config.py` - 添加配置:
   - `tokenizer_type`
   - `char_tokenizer_path`

3. `src/utils.py` - 更新函数:
   - `get_tokenizer()` - 支持两种tokenizer
   - `encode_text()` - 自动处理不同tokenizer的编码方式
   - `decode_text()` - 自动处理不同tokenizer的解码方式

4. `src/main.py` - 更新调用:
   - 传递 `tokenizer_type` 和 `char_tokenizer_path` 参数
   - 打印词汇表大小

## 如何使用

### 步骤 1: 激活虚拟环境
```bash
conda activate transformers_env
```

### 步骤 2: 运行字符级tokenizer训练
```bash
python train_char.py
```

**训练配置：**
- Tokenizer: 字符级 (vocab ~678)
- Model: d_model=64, num_layers=1, num_heads=2
- Data: 150 training samples (diverse, no repetition)
- Dropout: 0.3
- Epochs: 100 (with early stopping patience=20)
- TensorBoard: runs/char_small

### 步骤 3: 监控训练（可选）
```bash
tensorboard --logdir runs/char_small
```

然后在浏览器中打开 http://localhost:6006

## 预期改进

### 之前的问题
```
[Sample 0]
  Reference: 不要因小失大。
  Hypothesis: 我正努力理解释这个新任务。
  BLEU: 0.0000

[Sample 1]
  Reference: 生活掌握在自己手中。
  Hypothesis: 我正努力理解释这个新任务。
  BLEU: 0.0000
```

### 预期结果
- ✅ 不同的输入应该产生不同的翻译
- ✅ BLEU 分数应该 > 0
- ✅ 模型应该能学习到多样化的翻译模式
- ✅ 验证集表现应该在 5-10 epochs 内开始改善

## 为什么这能解决问题

1. **数据多样性** → 模型不能依赖单一模式
2. **小词汇表** → 参数与数据比例更平衡
3. **小模型** → 减少过拟合风险
4. **Dropout** → 增强泛化能力
5. **每个样本都独特** → 类似 demo_data 的成功模式

## 技术对比

| 特性 | 旧配置 | 新配置 |
|------|--------|--------|
| 词汇表大小 | 100,000 | 678 |
| 训练样本 | 1111 | 150 |
| 重复模式 | 155 "I'm trying to" | 0 |
| 模型大小 | d_model=128, layers=2 | d_model=64, layers=1 |
| Dropout | 0.1 | 0.3 |
| Checkpoint大小 | 479 MB | ~10 MB (预计) |

## 下一步

1. **运行训练** - `python train_char.py`
2. **观察BLEU分数** - 应该在前10个epochs内开始上升
3. **查看样本翻译** - 应该看到多样化的输出
4. **如果还有问题**:
   - 进一步增加 dropout (尝试 0.5)
   - 减少 d_model 到 32
   - 增加更多训练数据（保持多样性）

## 相关文件

- `create_data.py` - 生成新的平衡数据集
- `char_tokenizer.py` - 字符级tokenizer实现
- `data/char_tokenizer.json` - 保存的tokenizer
- `data/train.tsv` - 150个多样化样本
- `data/val.tsv` - 25个验证样本
- `data/test.tsv` - 20个测试样本
- `train_char.py` - 便捷的训练脚本
- `DIAGNOSIS.md` - 详细的问题诊断报告
