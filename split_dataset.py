#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
划分数据集为 train/val/test
默认比例：80% / 10% / 10%
"""
import random

# 读取原始数据集
input_file = "data/数据集_clean.tsv"
output_train = "data/train_clean.tsv"
output_val = "data/val_clean.tsv"
output_test = "data/test_clean.tsv"

# 读取所有数据
print(f"Reading data from {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# 打乱数据
random.seed(42)  # 设置随机种子以保证可重复性
random.shuffle(lines)

# 计算划分点
total = len(lines)
train_size = int(total * 0.8)
val_size = int(total * 0.1)
# test_size = total - train_size - val_size

# 划分数据
train_data = lines[:train_size]
val_data = lines[train_size:train_size + val_size]
test_data = lines[train_size + val_size:]

print(f"\nSplit results:")
print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
print(f"  Val:   {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
print(f"  Test:  {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")

# 保存数据
print(f"\nSaving to files...")
with open(output_train, 'w', encoding='utf-8') as f:
    f.writelines(train_data)
print(f"  Saved: {output_train}")

with open(output_val, 'w', encoding='utf-8') as f:
    f.writelines(val_data)
print(f"  Saved: {output_val}")

with open(output_test, 'w', encoding='utf-8') as f:
    f.writelines(test_data)
print(f"  Saved: {output_test}")

print("\nDone!")
