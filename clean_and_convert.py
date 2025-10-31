#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理和转换数据集脚本
1. 去除空行和格式错误的行
2. 统一编码为 UTF-8
3. 去除重复数据
"""
import sys


def clean_dataset(input_file, output_file):
    """
    清理数据集
    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    """
    print(f"Reading from: {input_file}")

    # 读取数据
    lines = []
    seen = set()  # 用于去重
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行
            if not line:
                skipped += 1
                continue

            # 检查格式
            try:
                src, tgt = line.split('\t')

                # 跳过空的源或目标
                if not src.strip() or not tgt.strip():
                    skipped += 1
                    continue

                # 去重
                if line in seen:
                    skipped += 1
                    continue

                seen.add(line)
                lines.append(line + '\n')

            except ValueError:
                print(f"Warning: Invalid format at line {line_num}: {line[:50]}...")
                skipped += 1
                continue

    # 保存清理后的数据
    print(f"\nWriting to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 统计
    print(f"\n{'='*60}")
    print(f"Cleaning completed!")
    print(f"{'='*60}")
    print(f"Total lines read:    {line_num}")
    print(f"Valid lines:         {len(lines)}")
    print(f"Skipped lines:       {skipped}")
    print(f"Duplicate lines:     {line_num - len(lines) - skipped}")
    print(f"Output file:         {output_file}")
    print(f"{'='*60}")


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("Usage: python clean_and_convert.py <input_file> <output_file>")
        print("\nExample:")
        print("  python clean_and_convert.py data/raw_data.tsv data/clean_data.tsv")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    clean_dataset(input_file, output_file)


if __name__ == '__main__':
    main()
