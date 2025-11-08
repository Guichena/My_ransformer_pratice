import torch
from args import get_args


class Config:
    def __init__(self):
        # 解析命令行参数
        args = get_args()

        # 模型参数
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # 训练参数
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.save_dir = args.save_dir
        self.log_interval = args.log_interval
        self.resume = args.resume
        self.save_best_only = args.save_best_only
        self.patience = args.patience
        self.max_eval_batches = args.max_eval_batches
        self.eval_interval = args.eval_interval
        self.skip_train_eval = args.skip_train_eval
        self.quick_eval = args.quick_eval
        self.tensorboard = args.tensorboard
        self.tensorboard_dir = args.tensorboard_dir

        # 数据参数
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.max_seq_len = args.max_seq_len
        self.use_demo_data = args.use_demo_data

        # Tokenizer 参数
        self.tokenizer_type = args.tokenizer_type
        self.char_tokenizer_path = args.char_tokenizer_path

        # 设置 device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            self.device = "cpu"

    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "Model Configuration:\n"
        config_str += f"  d_model: {self.d_model}\n"
        config_str += f"  d_ff: {self.d_ff}\n"
        config_str += f"  num_heads: {self.num_heads}\n"
        config_str += f"  num_layers: {self.num_layers}\n"
        config_str += f"  dropout: {self.dropout}\n"
        config_str += "\nTraining Configuration:\n"
        config_str += f"  batch_size: {self.batch_size}\n"
        config_str += f"  num_epochs: {self.num_epochs}\n"
        config_str += f"  learning_rate: {self.learning_rate}\n"
        config_str += f"  warmup_steps: {self.warmup_steps}\n"
        config_str += f"  device: {self.device}\n"
        config_str += f"  save_dir: {self.save_dir}\n"
        config_str += f"  log_interval: {self.log_interval}\n"
        config_str += f"  resume: {self.resume}\n"
        config_str += f"  save_best_only: {self.save_best_only}\n"
        config_str += f"  patience: {self.patience}\n"
        config_str += f"  max_eval_batches: {self.max_eval_batches}\n"
        config_str += f"  eval_interval: {self.eval_interval}\n"
        config_str += f"  skip_train_eval: {self.skip_train_eval}\n"
        config_str += f"  quick_eval: {self.quick_eval}\n"
        config_str += f"  tensorboard: {self.tensorboard}\n"
        config_str += f"  tensorboard_dir: {self.tensorboard_dir}\n"
        config_str += "\nData Configuration:\n"
        config_str += f"  train_file: {self.train_file}\n"
        config_str += f"  val_file: {self.val_file}\n"
        config_str += f"  max_seq_len: {self.max_seq_len}\n"
        config_str += f"  use_demo_data: {self.use_demo_data}\n"
        return config_str


# 创建全局配置实例
config = Config()
