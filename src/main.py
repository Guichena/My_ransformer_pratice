import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import Encoder, Decoder
from config import config
from optimizer import get_optimizer
from utils import (
    get_tokenizer,
    decode_text,
    evaluate_translations,
    get_demo_data,
    BOS_ID,
    EOS_ID,
    create_masks,
    create_padding_mask,
    create_tgt_mask,
)
from data import create_dataloader, load_data_from_file


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

    @property
    def device(self):
        return next(self.parameters()).device

    def translate(self, src, tokenizer, max_length=None):
        """
        翻译单个句子。
        参数:
        - src: 源序列
        - tokenizer: Tokenizer
        - max_length: 最长生成长度（默认使用 config.max_seq_len）
        返回:
        - 翻译结果
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            bos_token = BOS_ID
            eos_token = EOS_ID

            decode_max_length = max_length or min(config.max_seq_len, 50) or 50
            decode_max_length = max(2, decode_max_length)

            # 仅翻译第一条样本（batch 的第一个）
            single_src = src[0:1]

            # 将源序列移动到 device
            device = self.device
            single_src = single_src.to(device)

            # 创建源序列的 mask
            src_mask = create_padding_mask(single_src)

            # 编码源序列
            enc_output = self.encoder(single_src, src_mask)

            # 以 BOS token 初始化目标序列
            tgt = torch.tensor([[bos_token]], dtype=torch.long, device=device)

            # 自回归生成
            for _ in range(decode_max_length - 1):
                tgt_mask = create_tgt_mask(tgt)

                # 解码一步
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

                # 取下一步的 token
                next_token = output[:, -1].argmax(dim=-1, keepdim=True)

                # 将新 token 拼接到目标序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 若生成 EOS 则停止
                if next_token.item() == eos_token:
                    break

            # 将生成序列解码为文本
            translation = decode_text(tgt[0], tokenizer)
        if was_training:
            self.train()
        return translation


def train_step(model, optimizer, criterion, src, tgt_input, tgt_output):
    """
    执行单步训练。
    参数:
    - model: Transformer 模型
    - optimizer: Optimizer
    - criterion: 损失函数
    - src: 源序列
    - tgt_input: Decoder 输入序列
    - tgt_output: Decoder 目标输出
    返回:
    - 损失值
    """
    # 前向计算
    src_mask, tgt_mask = create_masks(src, tgt_input)

    output = model(src, tgt_input, src_mask, tgt_mask)

    # 计算损失
    output = output.view(-1, output.size(-1))
    # tgt_output = tgt_output.view(-1)
    tgt_output = tgt_output.reshape(-1)
    loss = criterion(output, tgt_output)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, tokenizer, eval_data, max_batches=None, quick_eval=False):
    """
    评估模型效果。
    参数:
    - model: Transformer 模型
    - tokenizer: Tokenizer
    - eval_data: 评估数据
    - max_batches: 最多评估的 batch 数
    - quick_eval: 快速评估模式（只评估前3个batch，展示5个样例）
    返回:
    - 平均 BLEU 分数
    """
    if eval_data is None:
        return 0.0

    if max_batches is not None and max_batches <= 0:
        return 0.0

    was_training = model.training
    model.eval()
    references = []
    hypotheses = []

    # 快速评估模式：只评估前3个batch
    eval_batches = 3 if quick_eval else max_batches

    with torch.no_grad():
        for batch_idx, (src, _, tgt_output) in enumerate(eval_data):
            if eval_batches is not None and batch_idx >= eval_batches:
                break

            src = src.to(model.device)
            batch_size = src.size(0)

            for sample_idx in range(batch_size):
                single_src = src[sample_idx : sample_idx + 1]
                translation = model.translate(single_src, tokenizer)
                reference = decode_text(tgt_output[sample_idx], tokenizer)

                references.append(reference)
                hypotheses.append(translation)

                # 打印前5条翻译样例
                if batch_idx == 0 and sample_idx < 5:
                    # 反解源文本以展示原句
                    source_text = decode_text(src[sample_idx], tokenizer)
                    print(f"  [Sample {sample_idx}]")
                    print(f"    原文:     {source_text}")
                    print(f"    标准翻译:  {reference}")
                    print(f"    生成翻译: {translation}")

    # 若没有有效翻译，返回 0
    if len(references) == 0:
        return 0.0

    bleu_score = evaluate_translations(references, hypotheses)

    # 快速评估模式提示
    if quick_eval:
        print(f"  (快速评估: 仅评估了 {len(references)} 条数据)")

    if was_training:
        model.train()

    return bleu_score


def train(model, optimizer, criterion, train_loader, val_loader, tokenizer):
    """
    Train the model.
    Parameters:
    - model: Transformer model
    - optimizer: Optimizer
    - criterion: Loss function
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - tokenizer: Tokenizer
    """
    # Create save directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize TensorBoard writer if enabled
    writer = None
    if config.tensorboard:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        writer = SummaryWriter(log_dir=config.tensorboard_dir)
        print(f"TensorBoard logging enabled. Log directory: {config.tensorboard_dir}")
        print(f"Run 'tensorboard --logdir={config.tensorboard_dir}' to view logs")

    # Initialize best BLEU score
    best_bleu = 0.0
    patience_counter = 0
    global_step = 0

    # Try initial evaluation to ensure everything works
    print("Initial evaluation...")
    try:
        # Evaluate on a small batch only
        initial_src, initial_tgt_input, initial_tgt_output = next(iter(train_loader))
        initial_src = initial_src[:1].to(model.device)
        initial_tgt_input = initial_tgt_input[:1].to(model.device)

        # Debug: Print tensor statistics
        print(f"DEBUG: src shape={initial_src.shape}, max={initial_src.max().item()}, min={initial_src.min().item()}")
        print(f"DEBUG: tgt_input shape={initial_tgt_input.shape}, max={initial_tgt_input.max().item()}, min={initial_tgt_input.min().item()}")
        print(f"DEBUG: Model encoder vocab_size={model.encoder.embedding.num_embeddings}")
        print(f"DEBUG: Model decoder vocab_size={model.decoder.embedding.num_embeddings}")

        src_mask, tgt_mask = create_masks(initial_src, initial_tgt_input)

        # Try forward pass
        with torch.no_grad():
            output = model(initial_src, initial_tgt_input, src_mask, tgt_mask)
        print("Initial evaluation successful!")
    except Exception as e:
        print(f"Initial evaluation failed: {e}")
        raise e

    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        for i, (src, tgt_input, tgt_output) in enumerate(train_loader):
            src = src.to(model.device)
            tgt_input = tgt_input.to(model.device)
            tgt_output = tgt_output.to(model.device)

            loss = train_step(model, optimizer, criterion, src, tgt_input, tgt_output)
            total_loss += loss
            global_step += 1

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/train_step", loss, global_step)

            # Print training information
            if (i + 1) % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}"
                )

                # Log average loss to TensorBoard
                if writer is not None:
                    writer.add_scalar("Loss/train_avg", avg_loss, global_step)

                total_loss = 0

        # Evaluate model (only every eval_interval epochs)
        if (epoch + 1) % config.eval_interval == 0:
            # Evaluate training set (unless skip_train_eval is set)
            if not config.skip_train_eval:
                print("Evaluating training set BLEU score...")
                train_bleu = evaluate(
                    model, tokenizer, train_loader, max_batches=config.max_eval_batches, quick_eval=config.quick_eval
                )
            else:
                print("Skipping training set evaluation (--skip_train_eval is set)")
                train_bleu = 0.0

            # Evaluate validation set
            val_bleu = 0.0
            if val_loader is not None:
                print("Evaluating validation set BLEU score...")
                val_bleu = evaluate(
                    model, tokenizer, val_loader, max_batches=config.max_eval_batches, quick_eval=config.quick_eval
                )
                if not config.skip_train_eval:
                    print(
                        f"Epoch {epoch + 1}/{config.num_epochs}, "
                        f"Train BLEU: {train_bleu:.4f}, "
                        f"Val BLEU: {val_bleu:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{config.num_epochs}, "
                        f"Val BLEU: {val_bleu:.4f}"
                    )
            else:
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Train BLEU: {train_bleu:.4f}"
                )

            # Log BLEU scores to TensorBoard
            if writer is not None:
                if not config.skip_train_eval:
                    writer.add_scalar("BLEU/train", train_bleu, epoch + 1)
                if val_loader is not None:
                    writer.add_scalar("BLEU/val", val_bleu, epoch + 1)
        else:
            # Skip evaluation for this epoch
            print(f"Epoch {epoch + 1}/{config.num_epochs}, Skipping evaluation (eval_interval={config.eval_interval})")
            train_bleu = 0.0
            val_bleu = 0.0

        # Save model
        if val_loader is None:
            improved = True
        else:
            improved = val_bleu > best_bleu

        if not config.save_best_only or improved:
            # Update best BLEU score
            if val_loader is not None:
                if val_bleu > best_bleu:
                    best_bleu = val_bleu
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_bleu": train_bleu,
                "val_bleu": val_bleu,
                "best_bleu": best_bleu,
            }
            torch.save(
                checkpoint,
                os.path.join(config.save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # Early stopping check (only enabled when validation set is available)
        if val_loader is not None and patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("TensorBoard logging finished")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load checkpoint.
    Parameters:
    - model: Transformer model
    - optimizer: Optimizer
    - checkpoint_path: Checkpoint file path
    Returns:
    - Starting epoch
    - Best BLEU score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_bleu"]


def main():
    # Print configuration
    print(config)

    # Get tokenizer
    print(f"Loading tokenizer (type: {config.tokenizer_type})...")
    tokenizer = get_tokenizer(config.tokenizer_type, config.char_tokenizer_path)
    print(f"Vocabulary size: {tokenizer.n_vocab}")

    # Prepare data
    if config.use_demo_data or not config.train_file:
        print("Using demo data")
        train_data, val_data, test_data = get_demo_data(tokenizer)
        train_loader = create_dataloader(
            train_data, config.batch_size, config.max_seq_len, tokenizer=tokenizer
        )
        val_loader = create_dataloader(val_data, config.batch_size, config.max_seq_len, tokenizer=tokenizer)
        test_loader = create_dataloader(
            test_data, config.batch_size, config.max_seq_len, tokenizer=tokenizer
        )
    else:
        print(f"Loading training data from file: {config.train_file}")
        train_loader = load_data_from_file(
            config.train_file, config.batch_size, config.max_seq_len, tokenizer=tokenizer
        )
        val_loader = (
            load_data_from_file(config.val_file, config.batch_size, config.max_seq_len, tokenizer=tokenizer)
            if config.val_file
            else None
        )
        test_loader = None

    # Create model
    model = Transformer(vocab_size=tokenizer.n_vocab).to(config.device)

    # Create optimizer and loss function
    optimizer = get_optimizer(
        model=model,
        model_size=config.d_model,
        factor=config.learning_rate,
        warmup=config.warmup_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Load checkpoint if specified
    if config.resume:
        load_checkpoint(model, optimizer, config.resume)

    # Train model
    train(model, optimizer, criterion, train_loader, val_loader, tokenizer)

    # Evaluate on test set if using demo data
    if test_loader is not None:
        test_bleu = evaluate(
            model, tokenizer, test_loader, max_batches=config.max_eval_batches
        )
        print(f"Test BLEU: {test_bleu:.4f}")


if __name__ == "__main__":
    main()
