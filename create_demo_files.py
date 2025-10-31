#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建demo数据文件
"""

# Demo训练数据（10个样本）
train_data = [
    ("Learning is the best reward.", "学习是旅途的意义。"),
    ("Knowledge is power.", "知识就是力量。"),
    ("Practice makes perfect.", "熟能生巧。"),
    ("Time is money.", "时间就是金钱。"),
    ("Where there is a will, there is a way.", "有志者事竟成。"),
    ("Actions speak louder than words.", "行动胜于言语。"),
    ("The early bird catches the worm.", "早起的鸟儿有虫吃。"),
    ("A journey of a thousand miles begins with a single step.", "千里之行，始于足下。"),
    ("Failure is the mother of success.", "失败是成功之母。"),
    ("Rome was not built in a day.", "罗马不是一天建成的。"),
]

# 验证数据（3个样本）
val_data = [
    ("Learning is the best reward.", "学习是旅途的意义。"),
    ("Knowledge is power.", "知识就是力量。"),
    ("Practice makes perfect.", "熟能生巧。"),
]

# 测试数据（3个样本）
test_data = [
    ("Time is money.", "时间就是金钱。"),
    ("Where there is a will, there is a way.", "有志者事竟成。"),
    ("Actions speak louder than words.", "行动胜于言语。"),
]

def write_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for src, tgt in data:
            f.write(f"{src}\t{tgt}\n")
    print(f"[OK] {filename}: {len(data)} lines")

if __name__ == "__main__":
    write_data("data/demo_train.tsv", train_data)
    write_data("data/demo_val.tsv", val_data)
    write_data("data/demo_test.tsv", test_data)
    print("\n[OK] Demo data files created!")
