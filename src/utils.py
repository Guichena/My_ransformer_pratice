import torch
import tiktoken
from typing import Tuple, List, Optional, Union
import collections
import math
import os
import sys

# 将父目录加入 sys.path，以便导入 CharTokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from char_tokenizer import CharTokenizer

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SPECIAL_TOKEN_COUNT = 3
TOKEN_OFFSET = SPECIAL_TOKEN_COUNT


_TOKENIZER: Optional[Union[tiktoken.Encoding, CharTokenizer]] = None
_TOKENIZER_TYPE: Optional[str] = None


def _get_or_create_tokenizer(tokenizer_type: str = "tiktoken", char_tokenizer_path: str = None):
    global _TOKENIZER, _TOKENIZER_TYPE

    if _TOKENIZER is None or _TOKENIZER_TYPE != tokenizer_type:
        if tokenizer_type == "char":
            if char_tokenizer_path and os.path.exists(char_tokenizer_path):
                _TOKENIZER = CharTokenizer.load(char_tokenizer_path)
            else:
                raise ValueError(f"Character tokenizer file not found: {char_tokenizer_path}")
        elif tokenizer_type == "tiktoken":
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        _TOKENIZER_TYPE = tokenizer_type

    return _TOKENIZER


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    创建 padding mask。
    参数:
    - seq: 输入序列
    返回:
    - padding mask
    """
    return (seq != PAD_ID).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    创建 look-ahead mask。
    参数:
    - size: 序列长度
    返回:
    - look-ahead mask
    """
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask


def create_tgt_mask(tgt):
    """创建目标序列的 mask（组合 look-ahead mask 与 padding mask）。"""
    seq_len = tgt.size(1)
    look_ahead = create_look_ahead_mask(seq_len).to(tgt.device)
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)

    padding_mask = create_padding_mask(tgt)
    padding_mask = padding_mask.expand(-1, -1, seq_len, -1)

    return (~look_ahead) & padding_mask


def create_masks(src, tgt):
    """
    创建源序列与目标序列的 masks。
    参数:
    - src: 源序列，形状 [batch_size, seq_len]
    - tgt: 目标序列，形状 [batch_size, seq_len]
    返回:
    - src_mask: 源序列 mask，形状 [batch_size, 1, 1, seq_len]
    - tgt_mask: 目标序列 mask，形状 [batch_size, 1, seq_len, seq_len]
    """
    src_mask = create_padding_mask(src)
    tgt_mask = create_tgt_mask(tgt)
    return src_mask, tgt_mask


def get_tokenizer(tokenizer_type: str = "tiktoken", char_tokenizer_path: str = None):
    return _get_or_create_tokenizer(tokenizer_type, char_tokenizer_path)


def encode_text(
    text: str, tokenizer, max_seq_len: Optional[int] = None
) -> torch.Tensor:
    # 对于 CharTokenizer，不需要加 TOKEN_OFFSET（其 vocab 已从 4 开始）
    if isinstance(tokenizer, CharTokenizer):
        content_tokens = tokenizer.encode(text)
    else:
        # 对于 tiktoken，需要加 offset
        content_tokens = [token + TOKEN_OFFSET for token in tokenizer.encode(text)]

    tokens = [BOS_ID] + content_tokens + [EOS_ID]
    if max_seq_len is not None and max_seq_len > 0:
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            if tokens[-1] != EOS_ID:
                tokens[-1] = EOS_ID
    return torch.tensor(tokens, dtype=torch.long)


def decode_text(tokens: torch.Tensor, tokenizer) -> str:
    if tokens.dim() > 1:
        tokens = tokens.squeeze(0)
    tokens = tokens.cpu().numpy().tolist()
    if EOS_ID in tokens:
        tokens = tokens[: tokens.index(EOS_ID)]
    tokens = [t for t in tokens if t not in [PAD_ID, BOS_ID]]

    # 对于 CharTokenizer，直接解码无需 offset
    if isinstance(tokenizer, CharTokenizer):
        return tokenizer.decode(tokens)
    else:
        # 对于 tiktoken，需要减去 offset
        content_tokens = [t - TOKEN_OFFSET for t in tokens if t >= TOKEN_OFFSET]
        return tokenizer.decode(content_tokens)


def calculate_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    计算 BLEU 分数。
    参数:
    - reference: 参考翻译
    - hypothesis: 模型生成的翻译
    - max_n: n-gram 的最大长度
    返回:
    - [0, 1] 区间内的 BLEU 分数
    """
    # 将文本按字符切分（对中文更友好）
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    # 若生成翻译为空，直接返回 0
    if len(hyp_tokens) == 0:
        return 0.0

    # 计算各阶 n-gram 的 precision
    precisions = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        # 统计参考翻译中的 n-grams
        ref_ngrams = collections.Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i : i + n])
            ref_ngrams[ngram] += 1

        # 统计生成翻译中的 n-grams
        hyp_ngrams = collections.Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i : i + n])
            hyp_ngrams[ngram] += 1

        # 计算匹配到的 n-grams 数量
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        # 计算 precision
        precision = matches / max(1, len(hyp_tokens) - n + 1)
        precisions.append(precision)

    # 计算 brevity penalty（长度惩罚）
    if len(hyp_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        brevity_penalty = 1.0

    # 汇总计算 BLEU 分数
    if any(p > 0 for p in precisions):
        s = math.log(brevity_penalty)
        s += sum(math.log(p) if p > 0 else float("-inf") for p in precisions) / len(
            precisions
        )
        bleu = math.exp(s)
    else:
        bleu = 0.0

    return bleu


def evaluate_translations(references: List[str], hypotheses: List[str]) -> float:
    """
    对一组翻译计算平均 BLEU 分数。
    参数:
    - references: 参考翻译列表
    - hypotheses: 模型生成的翻译列表
    返回:
    - 平均 BLEU 分数
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of reference and generated translations must be equal")

    total_bleu = 0.0
    for ref, hyp in zip(references, hypotheses):
        total_bleu += calculate_bleu(ref, hyp)

    return total_bleu / len(references)


def get_demo_data(
    tokenizer: tiktoken.Encoding,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """
    获取 demo 的训练/验证/测试数据。
    参数:
    - tokenizer: Tokenizer
    返回:
    - (train_data, val_data, test_data) 的三元组；每个元素为 (source_text, target_text)
    """
    # 训练数据
    train_data = [
        ("Learning is the best reward.", "学习是旅途的意义。"),
        ("Knowledge is power.", "知识就是力量。"),
        ("Practice makes perfect.", "熟能生巧。"),
        ("Time is money.", "时间就是金钱。"),
        ("Where there is a will, there is a way.", "有志者事竟成。"),
        ("Actions speak louder than words.", "行动胜于言语。"),
        ("The early bird catches the worm.", "早起的鸟儿有虫吃。"),
        (
            "A journey of a thousand miles begins with a single step.",
            "千里之行，始于足下。",
        ),
        ("Failure is the mother of success.", "失败是成功之母。"),
        ("Rome was not built in a day.", "罗马不是一天建成的。"),
    ]

    # 验证数据
    val_data = [
        ("All roads lead to Rome.", "条条大路通罗马。"),
        ("Better late than never.", "亡羊补牢，为时未晚。"),
        ("Every cloud has a silver lining.", "黑暗中总有一线光明。"),
        ("A friend in need is a friend indeed.", "患难见真情。"),
        ("Honesty is the best policy.", "诚实是最好的策略。"),
    ]

    # 测试数据
    test_data = [
        ("The grass is always greener on the other side.", "这山望着那山高。"),
        ("Don't put all your eggs in one basket.", "不要把所有鸡蛋放在一个篮子里。"),
        ("When in Rome, do as the Romans do.", "入乡随俗。"),
        ("A penny saved is a penny earned.", "省一分就是赚一分。"),
        ("Birds of a feather flock together.", "物以类聚，人以群分。"),
    ]

    return train_data, val_data, test_data


def save_data_to_file(data: List[Tuple[str, str]], file_path: str) -> None:
    """
    将数据保存到文件。
    参数:
    - data: (source_text, target_text) 的列表
    - file_path: 文件路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for src, tgt in data:
            f.write(f"{src}\t{tgt}\n")
