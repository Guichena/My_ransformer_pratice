import argparse


def get_args():
    """
    解析命令行参数。
    返回:
    - 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Transformer model training arguments")

    # 模型参数
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="模型维度",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="前馈网络维度",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="注意力头数量",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Encoder 与 Decoder 层数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout 比例",
    )

    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch 大小",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="训练的 epoch 数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="学习率缩放因子",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=60,
        help="warmup 步数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="训练设备",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="模型 checkpoint 保存目录",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="日志打印的 batch 间隔",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的 checkpoint 路径",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="仅在验证集 BLEU 提升时保存 checkpoint",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early Stopping 的耐心（单位: epoch）",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="评估的最大 batch 数（None 表示全部）",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="每 N 个 epoch 评估一次（默认 1，表示每个 epoch）",
    )
    parser.add_argument(
        "--skip_train_eval",
        action="store_true",
        help="跳过训练集评估（仅评估验证集）",
    )
    parser.add_argument(
        "--quick_eval",
        action="store_true",
        help="快速评估模式（只评估前3个batch约96条数据，大幅加快训练速度）",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="启用 TensorBoard 日志",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="TensorBoard 日志目录",
    )

    # 数据参数
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="训练数据文件路径",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="验证数据文件路径",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="最大序列长度",
    )
    parser.add_argument(
        "--use_demo_data",
        action="store_true",
        help="使用 demo 数据集",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="char",
        choices=["char", "tiktoken"],
        help="使用的 tokenizer 类型：'char'（字符级，小词表）或 'tiktoken'（BPE，大词表）",
    )
    parser.add_argument(
        "--char_tokenizer_path",
        type=str,
        default="data/char_tokenizer.json",
        help="字符级 tokenizer 文件路径",
    )

    return parser.parse_args()
