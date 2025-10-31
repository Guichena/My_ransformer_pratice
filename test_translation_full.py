#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试翻译 - 使用完整数据集训练的模型
输入英文，输出中文
"""
import sys
import torch
import glob
import os

# 在导入config之前设置命令行参数
sys.argv = ['test_translation_full.py',
            '--tokenizer_type', 'char',
            '--char_tokenizer_path', 'data/char_tokenizer.json']

sys.path.insert(0, 'src')

from main import Transformer
from utils import get_tokenizer, encode_text, decode_text, create_masks
from config import config

def translate(model, tokenizer, text, device, max_length=100):
    """
    翻译单个句子
    """
    model.eval()

    # 编码输入
    src = encode_text(text, tokenizer, max_seq_len=512).unsqueeze(0).to(device)

    # 生成翻译
    with torch.no_grad():
        # 从BOS开始
        tgt = torch.tensor([[1]], dtype=torch.long).to(device)  # BOS_ID = 1

        for _ in range(max_length):
            # 创建mask
            src_mask, tgt_mask = create_masks(src, tgt)

            # 前向传播
            output = model(src, tgt, src_mask, tgt_mask)

            # 获取最后一个位置的预测
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # 拼接
            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果生成了EOS，停止
            if next_token.item() == 2:  # EOS_ID = 2
                break

    # 解码
    translation = decode_text(tgt.squeeze(0), tokenizer)
    return translation

def main():
    print("="*60)
    print("Transformer Translation Test (Full Dataset Model)")
    print("="*60)

    # 加载tokenizer
    print(f"\nLoading tokenizer: {config.char_tokenizer_path}")
    tokenizer = get_tokenizer(config.tokenizer_type, config.char_tokenizer_path)
    print(f"Vocabulary size: {tokenizer.n_vocab}")

    # 查找最新的checkpoint
    checkpoint_files = glob.glob("checkpoints/*.pt")
    # 过滤出大文件（>10MB，完整数据集模型）
    checkpoint_files = [f for f in checkpoint_files if os.path.getsize(f) > 10*1024*1024]

    if not checkpoint_files:
        print("\nError: No trained model found!")
        print("Please run: train_full_dataset.bat")
        return

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")

    # 加载checkpoint并推断模型配置
    checkpoint = torch.load(latest_checkpoint, map_location=config.device)

    # 从checkpoint推断模型配置
    state_dict = checkpoint['model_state_dict']
    inferred_d_model = state_dict['encoder.embedding.weight'].shape[1]
    inferred_vocab_size = state_dict['encoder.embedding.weight'].shape[0]
    inferred_num_layers = len([k for k in state_dict.keys() if 'encoder.layers' in k and 'self_attn.w_q.weight' in k])
    inferred_d_ff = state_dict['encoder.layers.0.feed_forward.w_1.weight'].shape[0]
    # num_heads需要从注意力层推断
    inferred_num_heads = 8 if inferred_d_model >= 512 else (4 if inferred_d_model >= 256 else 2)

    print(f"\nInferred model configuration from checkpoint:")
    print(f"  vocab_size: {inferred_vocab_size}")
    print(f"  d_model: {inferred_d_model}")
    print(f"  num_layers: {inferred_num_layers}")
    print(f"  d_ff: {inferred_d_ff}")
    print(f"  num_heads: {inferred_num_heads} (estimated)")

    # 更新config以匹配checkpoint的模型配置
    config.d_model = inferred_d_model
    config.num_layers = inferred_num_layers
    config.d_ff = inferred_d_ff
    config.num_heads = inferred_num_heads

    # 创建模型（使用推断的配置）
    print("\nCreating model with inferred configuration...")
    model = Transformer(vocab_size=inferred_vocab_size).to(config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded! (Epoch: {checkpoint.get('epoch', 'unknown')}, Val BLEU: {checkpoint.get('val_bleu', 0):.4f})")

    # 测试翻译
    print("\n" + "="*60)
    print("Translation Test")
    print("="*60)

    # 测试一些常见句子
    test_sentences = [
        "Hello, how are you?",
        "I love learning new things.",
        "The weather is nice today.",
        "What time is it?",
        "Thank you very much.",
        "I want to go to the park.",
        "She is reading a book.",
        "We are studying English.",
    ]

    for i, sent in enumerate(test_sentences, 1):
        translation = translate(model, tokenizer, sent, config.device)
        print(f"\n{i}. English: {sent}")
        print(f"   Chinese: {translation}")

    # 交互式翻译
    print("\n" + "="*60)
    print("Interactive Translation Mode")
    print("="*60)
    print("Enter English sentences to translate (or 'quit' to exit)")
    print()

    while True:
        try:
            text = input("English: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue

            translation = translate(model, tokenizer, text, config.device)
            print(f"Chinese: {translation}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nGoodbye!")

if __name__ == '__main__':
    main()
