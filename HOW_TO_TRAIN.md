# 如何运行新数据集训练

## 方法 1：在终端中手动运行（推荐）

打开终端，依次执行以下命令：

```bash
# 1. 激活 conda 环境
conda activate transformers_env

# 2. 进入项目目录
cd "d:\新建文件夹\My_Own_Transformers"

# 3. 运行训练
python src/main.py --train_file data/train.tsv --val_file data/val.tsv --tensorboard --device cuda --num_epochs 20 --batch_size 16 --log_interval 5
```

## 方法 2：使用批处理脚本

双击运行 `run_training.bat` 文件

## 训练参数说明

- `--train_file data/train.tsv`: 使用 98 条训练数据
- `--val_file data/val.tsv`: 使用 25 条验证数据
- `--tensorboard`: 启用 TensorBoard 可视化
- `--device cuda`: 使用 GPU 加速
- `--num_epochs 20`: 训练 20 个 epoch
- `--batch_size 16`: 批次大小 16
- `--log_interval 5`: 每 5 个 batch 打印一次日志

## 查看 TensorBoard

训练开始后，在另一个终端运行：

```bash
conda activate transformers_env
tensorboard --logdir=runs
```

然后在浏览器打开 http://localhost:6006

## 预期效果

随着训练进行，你应该看到：

```
Epoch 1/20, Train BLEU: 0.05, Val BLEU: 0.03
Epoch 2/20, Train BLEU: 0.12, Val BLEU: 0.08
Epoch 3/20, Train BLEU: 0.18, Val BLEU: 0.14
...
Epoch 20/20, Train BLEU: 0.65, Val BLEU: 0.52
```

BLEU 分数应该逐渐提升，表示模型正在学习英译中的翻译能力！

## 训练完成后

模型检查点会保存在 `checkpoints/` 目录：
- `checkpoint_epoch_1.pt`
- `checkpoint_epoch_2.pt`
- ...
- `checkpoint_epoch_20.pt`

每个检查点包含：
- 模型权重
- 优化器状态
- 训练和验证 BLEU 分数
