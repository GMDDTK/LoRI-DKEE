"""
finetune_llama_classification.py

用途：
  - 对本地 LLaMA-3.2-1B（路径示例：G:\pretrain_models\0Large Language Models\Llama-3.2-1B）
    使用 QLoRA（4-bit + LoRA）做二分类（句子是否为“创新句”1/0）。

说明：
  - 数据：E:\PythonProject\科学计量学谱系\知识元分类\classification2 下的 train.txt/val.txt/test.txt
    格式：每行一条，文本<TAB>标签（0 或 1）
  - 输出：保存 LoRA adapter 到 output_dir（默认 ./lora_out）
  - 运行：python finetune_llama_classification.py
"""

import os
import sys
import math
import time
import random
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict
import functools

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# 基本配置（根据你卡的显存可以调）
# ----------------------------
DEFAULTS = {
    "model_path": r"G:\pretrain_models\0Large Language Models\Llama-3.2-3B",
    "data_dir": r"E:\PythonProject\科学计量学谱系\知识元分类\classification2",
    "output_dir": "./lora_out",
    "micro_batch_size": 4,             # 每个 step 的微批（尽量不超显存；不行就降到 4 或 2）
    "gradient_accumulation_steps": 4,  # 与 micro_batch_size 一起决定有效 batch
    "max_epochs": 3,
    "lr": 2e-4,                        # LoRA 常用范围 1e-4 - 3e-4
    "weight_decay": 0.0,
    "warmup_steps": 50,
    "max_length": 256,
    "seed": 42,
    "lora_r": 8,                       # 对于 1B，r=4..16 常用；r 越小内存越低但表示能力有限
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "save_every_epoch": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ----------------------------
# 简单日志
# ----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ----------------------------
# 数据处理
# ----------------------------
def read_tsv(file_path: str) -> List[Dict]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # 假设文本中可能包含 TAB，但格式为 "文本<TAB>标签"
            if "\t" not in line:
                logger.warning(f"Line {i+1} in {file_path} does not contain TAB; skipping.")
                continue
            text, label = line.rsplit("\t", 1)
            label = label.strip()
            if label not in {"0", "1"}:
                logger.warning(f"Line {i+1} label not 0/1: '{label}'; skipping.")
                continue
            items.append({"text": text.strip(), "label": int(label)})
    return items


def make_prompt_and_target(text: str, label: int) -> (str, str):
    # 设计 prompt：尽量短、清晰，生成“0”或“1”作为答案
    prompt = (
        "Classify whether the following academic sentence is an *innovative* sentence.\n\n"
        "Sentence: " + text.strip() + "\n"
        "Answer (1 = innovative, 0 = not):"
    )
    target = " " + str(label)  # 前面留一个空格确保 tokenizer 清楚分割
    return prompt, target


class CLSDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        prompt, target = make_prompt_and_target(it["text"], it["label"])
        # 先 tokenize full_text 再找到 label 开始位置来做 labels mask
        # 注意：使用 add_special_tokens=True 以确保与模型兼容
        full = prompt + target
        enc_full = self.tokenizer(full, truncation=True, max_length=self.max_length, add_special_tokens=True)
        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length, add_special_tokens=True)

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full.get("attention_mask", [1]*len(input_ids))

        # label: 只有 target 部分计算 loss，其余为 -100
        labels = [-100] * len(enc_full["input_ids"])
        prompt_len = len(enc_prompt["input_ids"])
        # 允许 target 被截断成多 token，labels 从 prompt_len 开始全部是真实 token id
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }


def collate_fn_with_pad(batch, pad_token_id):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }


# ----------------------------
# 训练 / 评估 主函数
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_save_peft(model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # model 这里是 PeftModel（get_peft_model 的结果）
    logger.info(f"Saving LoRA adapter to {out_dir} (only adapter weights).")
    model.save_pretrained(out_dir)


def evaluate_generation(model, tokenizer, items, device, max_length=64, batch_size=32):
    """
    用 generate 在验证集上预测 label（1/0），计算 accuracy/precision/recall/f1
    注意：这里用了简单的文本解析策略提取 0/1
    """
    model.eval()
    preds = []
    truths = []
    dataloader = DataLoader(items, batch_size=batch_size, shuffle=False)
    for batch_items in dataloader:
        for it in batch_items:
            prompt, _ = make_prompt_and_target(it["text"], 0)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # out_ids 包含 prompt + generated tokens
            gen_part = out_ids[0, input_ids.shape[1]:].cpu().numpy().tolist()
            gen_text = tokenizer.decode(gen_part, skip_special_tokens=True).strip().lower()
            # 解析 0/1
            pred = parse_label_from_generated(gen_text)
            preds.append(pred)
            truths.append(it["label"])
    model.train()
    acc = accuracy_score(truths, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(truths, preds, average='binary', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def parse_label_from_generated(text: str) -> int:
    # 找首个独立的 0/1 token
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))
    if "yes" in text or "innov" in text or "1" in text:
        return 1
    if "no" in text or "not" in text or "0" in text:
        return 0
    # 保守返回 0
    return 0


def main(args):
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    device = cfg["device"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_items = read_tsv(str(Path(cfg["data_dir"]) / "train.txt"))
    val_items = read_tsv(str(Path(cfg["data_dir"]) / "val.txt"))
    test_items = read_tsv(str(Path(cfg["data_dir"]) / "test.txt"))

    train_ds = CLSDataset(train_items, tokenizer, cfg["max_length"])
    val_ds = CLSDataset(val_items, tokenizer, cfg["max_length"])
    test_ds = CLSDataset(test_items, tokenizer, cfg["max_length"])

    # 🔑 修复点：用 functools.partial 传递 pad_token_id，避免 lambda
    collate = functools.partial(collate_fn_with_pad, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["micro_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,       # Windows 推荐 0
        pin_memory=True
    )
    val_loader_for_eval = val_items  # evaluate_generation 采用简单逐条生成

    # --------- BitsAndBytesConfig & 模型加载（4-bit） ---------
    logger.info("Preparing BitsAndBytesConfig (4-bit NF4, double quant) and loading model in 4-bit.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    logger.info("Loading model (this may take a while).")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # 推荐关闭 use_cache 以避免生成时占用额外显存
    model.config.use_cache = False

    # --------- prepare model for k-bit training + apply LoRA ---------
    logger.info("Preparing model for k-bit training (PEFT.prepare_model_for_kbit_training).")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    logger.info("Applying LoRA (get_peft_model).")
    model = get_peft_model(model, lora_config)

    # 打印可训练参数统计
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f"Trainable params: {trainable_params} / {all_params} ({100*trainable_params/all_params:.4f}%)")

    # --------- 优化器 / scheduler / amp ---------
    optimizer_grouped_parameters = [
        {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": cfg["weight_decay"]}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg["lr"])

    total_update_steps = math.ceil(len(train_loader) * cfg["max_epochs"] / cfg["gradient_accumulation_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_update_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    # --------- 训练循环 ---------
    global_step = 0
    best_val_f1 = 0.0
    model.train()
    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # batch tensors
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向 + 反向（mixed precision）
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * cfg["gradient_accumulation_steps"]

            # 监控（每若干 step 打印）
            if (step + 1) % (cfg["gradient_accumulation_steps"] * 50) == 0:
                logger.info(f"Epoch {epoch} step {step+1}/{len(train_loader)} loss={epoch_loss/(step+1):.4f}")

        t1 = time.time()
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished in {(t1-t0):.1f}s, avg_loss={avg_epoch_loss:.6f}")

        # --------- 验证（生成方式）---------
        try:
            val_metrics = evaluate_generation(model, tokenizer, val_ds, device, max_length=64, batch_size=32)
            logger.info(f"Validation metrics after epoch {epoch}: {val_metrics}")
        except Exception as e:
            logger.warning(f"Validation generation failed: {e}")
            val_metrics = {}

        # 保存 adapter
        if cfg["save_every_epoch"]:
            out_dir = os.path.join(cfg["output_dir"], f"epoch{epoch}")
            safe_save_peft(model, out_dir)

        # Early save best f1
        if "f1" in val_metrics and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            safe_save_peft(model, os.path.join(cfg["output_dir"], "best"))

    # 最终保存
    safe_save_peft(model, os.path.join(cfg["output_dir"], "final"))
    logger.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULTS["model_path"])
    parser.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])
    parser.add_argument("--micro_batch_size", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max_length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--lora_r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=DEFAULTS["lora_dropout"])
    args = parser.parse_args()
    main(args)
