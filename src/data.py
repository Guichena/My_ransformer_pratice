import os
import torch
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from typing import List, Tuple
from utils import encode_text, get_tokenizer, PAD_ID


class TranslationDataset(torch.utils.data.Dataset):
    """
    翻译数据集类。
    参数:
    - data: (source_text, target_text) 的列表
    - tokenizer: Tokenizer
    - max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for src_text, tgt_text in data:
            src_tokens = encode_text(src_text, tokenizer, max_seq_len)
            tgt_tokens = encode_text(tgt_text, tokenizer, max_seq_len)
            if src_tokens.numel() < 2 or tgt_tokens.numel() < 2:
                # 边界情况：若长度不足（少于 BOS/EOS），跳过该样本
                continue
            self.data.append((src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_sequence(sequences):
    """使用 PAD_ID 将序列填充到相同长度。"""
    return torch_pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)


def create_dataloader(
    data: List[Tuple[str, str]],
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
    tokenizer=None,  # Accept tokenizer as parameter
) -> torch.utils.data.DataLoader:
    """
    创建 DataLoader。
    参数:
    - data: (source_text, target_text) 的列表
    - batch_size: batch 大小
    - max_seq_len: 最大序列长度
    - shuffle: 是否打乱数据
    - tokenizer: Tokenizer 实例（为 None 时使用默认）
    返回:
    - DataLoader 实例
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    dataset = TranslationDataset(data, tokenizer, max_seq_len)

    def collate_fn(batch):
        src_batch = pad_sequence([item[0] for item in batch])
        tgt_batch = pad_sequence([item[1] for item in batch])

        if tgt_batch.size(1) < 2:
            raise ValueError(
                "目标序列长度至少为 2（包含 BOS 和 EOS）"
            )

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        return src_batch, tgt_input, tgt_output

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def load_data_from_file(
    file_path: str,
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
    tokenizer=None,  # Accept tokenizer as parameter
) -> torch.utils.data.DataLoader:
    """
    从文件加载数据并创建 DataLoader。
    参数:
    - file_path: 文件路径
    - batch_size: batch 大小
    - max_seq_len: 最大序列长度
    - shuffle: 是否打乱数据
    - tokenizer: Tokenizer 实例（为 None 时使用默认）
    返回:
    - DataLoader 实例
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                src, tgt = line.split("\t")
                data.append((src, tgt))
            except ValueError:
                print(f"跳过无效行: {line}")

    return create_dataloader(data, batch_size, max_seq_len, shuffle, tokenizer)
