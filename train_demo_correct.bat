@echo off
echo ======================================================================
echo Demo Data Training - Correct Setup
echo ======================================================================
echo.

REM 步骤1：创建demo数据文件
echo Step 1: Creating demo data files...
python create_demo_files.py
echo.

REM 步骤2：为demo data构建专用tokenizer
echo Step 2: Building tokenizer from demo data only...
python -c "from char_tokenizer import build_tokenizer_from_files; build_tokenizer_from_files('data/demo_train.tsv', 'data/demo_val.tsv', 'data/char_tokenizer_demo.json')"
echo.

REM 步骤3：训练
echo ======================================================================
echo Step 3: Starting training...
echo ======================================================================
echo.

call conda activate transformers_env && python src/main.py --train_file data/demo_train.tsv --val_file data/demo_val.tsv --tokenizer_type char --char_tokenizer_path data/char_tokenizer_demo.json --d_model 64 --d_ff 128 --num_heads 2 --num_layers 1 --dropout 0.1 --batch_size 4 --num_epochs 200 --learning_rate 1.0 --warmup_steps 50 --log_interval 10 --patience 200 --save_best_only --eval_interval 1 --tensorboard --tensorboard_dir runs/demo_correct

echo.
echo ======================================================================
echo Training completed!
echo ======================================================================
pause
