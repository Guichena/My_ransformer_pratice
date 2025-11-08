@echo off
REM 使用完整 train_clean 数据集训练 Transformer 模型
REM 数据集大小：18746 训练样本
REM 推荐配置：中等模型 + 字符级分词

REM 切换到项目目录
cd /d D:\新建文件夹\My_Own_Transformers

REM 激活 conda 环境
call conda activate transformers_env

echo ============================================================
echo Training Transformer with Full Dataset
echo ============================================================
echo.
echo Current Directory: %CD%
echo Conda Environment: %CONDA_DEFAULT_ENV%
echo.
echo Dataset Info:
echo   Training: 18,746 samples
echo   Validation: 2,343 samples
echo   Test: 2,344 samples
echo   Vocabulary: 2,694 characters
echo.
echo Model Configuration:
echo   d_model: 256 (medium size for ~20k samples)
echo   num_layers: 4 (balanced depth)
echo   num_heads: 8
echo   d_ff: 1024
echo   dropout: 0.1
echo.
echo Training Configuration:
echo   batch_size: 32
echo   num_epochs: 50
echo   learning_rate: 1.0 (with warmup)
echo   warmup_steps: 4000
echo   early_stopping: patience=5
echo   TensorBoard: enabled
echo.
echo ============================================================
pause

python src/main.py ^
    --train_file data/train_clean.tsv ^
    --val_file data/val_clean.tsv ^
    --tokenizer_type char ^
    --char_tokenizer_path data/char_tokenizer.json ^
    --d_model 256 ^
    --num_layers 4 ^
    --num_heads 8 ^
    --d_ff 1024 ^
    --dropout 0.1 ^
    --batch_size 32 ^
    --num_epochs 50 ^
    --learning_rate 1.0 ^
    --warmup_steps 4000 ^
    --patience 5 ^
    --save_best_only ^
    --tensorboard ^
    --tensorboard_dir runs/full_dataset ^
    --device cuda

echo.
echo ============================================================
echo Training completed!
echo.
echo Next steps:
echo   1. Check TensorBoard: tensorboard --logdir=runs/full_dataset
echo   2. Test the model: python test_translation_full.py
echo ============================================================
pause
