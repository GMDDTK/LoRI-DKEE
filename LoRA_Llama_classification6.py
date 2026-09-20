"""
finetune_llama_classification6.py

用途：
  - 对本地 LLaMA 使用 QLoRA（4-bit + LoRA）做六分类（0-5）。
  - 数据：每行一条，文本<TAB>标签（0/1/2/3/4/5）
    例：E:\PythonProject\科学计量学谱系\知识元分类\classification6\train.txt

重要设计说明（简短）：
  - Prompt 非常简洁，明确要求“只返回单个数字 0-5（no extra words）”，并列出类别映射。
  - max_length 默认 128（prompt+label 总长），micro_batch_size 默认 4，适合 16GB 显存的保守设定。
"""

import os
import math
import time
import random
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
import functools

import torch
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
# 基本配置（可按你显存微调）
# ----------------------------
DEFAULTS = {
    "model_path": r"G:\pretrain_models\0Large Language Models\Llama-3.2-3B",
    "data_dir": r"E:\PythonProject\科学计量学谱系\知识元分类\classification6",
    "output_dir": "./lora_out_classification6",
    "micro_batch_size": 4,             # 为 16GB 显存做保守设置（可酌情调小/调大）
    "gradient_accumulation_steps": 4,  # 与 micro_batch_size 共同决定有效 batch
    "max_epochs": 3,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "warmup_steps": 50,
    "max_length": 256,
    "seed": 42,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "save_every_epoch": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # 生成时参数
    "gen_max_new_tokens": 6,
    "eval_batch_size": 16,
}

# 类别映射（供 prompt & 评估使用）
LABEL_MAP = {
    0: "None/No classification (negative)",
    1: "Objective & Scope",
    2: "Data & Materials",
    3: "Methodology",
    4: "Results & Findings",
    5: "Discussion & Conclusion"
}

# ----------------------------
# 日志
# ----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ----------------------------
# 读数据（允许行内有 TAB，但最后一段为 label）
# ----------------------------
def read_tsv(file_path: str) -> List[Dict]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                logger.warning(f"Line {i+1} in {file_path} does not contain TAB; skipping.")
                continue
            text, label = line.rsplit("\t", 1)
            label = label.strip()
            if label not in {"0", "1", "2", "3", "4", "5"}:
                logger.warning(f"Line {i+1} label not in 0-5: '{label}'; skipping.")
                continue
            items.append({"text": text.strip(), "label": int(label)})
    logger.info(f"Read {len(items)} items from {file_path}")
    return items


# ----------------------------
# Prompt 设计（短、明确，只返回数字 0-5）
# ----------------------------
def make_prompt_and_target(text: str, label: int) -> Tuple[str, str]:
    """
    Prompt 目标：
      - 简洁明了地告诉模型只输出单个数字 0-5。
      - 包含类别映射但尽量短。
    返回 (prompt, target)：
      - prompt: 给模型的完整输入（不包含 target token）
      - target: 要模型输出的标签（带一个前导空格，方便 tokenizer 分割）
    """
    # 构建映射行（压缩在一行以节省 token）
    mapping = "; ".join([f"{k}={v}" for k, v in LABEL_MAP.items()])
    prompt = (
        "Classify the following academic sentence into one of six categories below. "
        "Reply with ONLY the single digit 0-5 (no extra words).\n"
        f"Categories: {mapping}\n\n"
        "Sentence: " + text.strip() + "\n"
        "Answer (single digit 0-5):"
    )
    target = " " + str(label)
    return prompt, target


# ----------------------------
# Dataset（构造 input_ids 与 labels，只对 target 部分计算 loss）
# ----------------------------
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
        full = prompt + target

        enc_full = self.tokenizer(full, truncation=True, max_length=self.max_length, add_special_tokens=True)
        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length, add_special_tokens=True)

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full.get("attention_mask", [1] * len(input_ids))

        # labels: prompt 部分为 -100，target 部分为真实 id（可能是多个 token）
        labels = [-100] * len(input_ids)
        prompt_len = len(enc_prompt["input_ids"])
        # 如果 prompt_len >= len(input_ids) 意味着 target 被截断掉了 -> 此样本将不会有 label（全为 -100），训练时不会对它计 loss
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": it["text"],
            "label": it["label"],
        }


def collate_fn_with_pad(batch, pad_token_id):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_token_id).long()

    meta_texts = [x.get("text", "") for x in batch]
    meta_labels = [x.get("label", -1) for x in batch]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "meta_texts": meta_texts,
        "meta_labels": meta_labels
    }


# ----------------------------
# 评估：batch 生成并解析 0-5
# ----------------------------
def parse_label_from_generated(text: str) -> int:
    """
    从生成文本中提取 0-5 的第一个独立数字。若无法识别，则保守返回 0（无分类负样本）。
    """
    text_low = text.lower()
    # 首先找独立数字 0-5
    m = re.search(r"\b([0-5])\b", text_low)
    if m:
        return int(m.group(1))
    # 其次尝试匹配单词线索（宽松）
    for k, v in LABEL_MAP.items():
        key = v.split()[0].lower()  # 比如 "Objective" or "Data"
        if key in text_low:
            return k
    # 保守返回 0
    return 0


def evaluate_generation(model, tokenizer, items: List[Dict], device, cfg) -> Dict:
    """
    对 items（list of {'text':..., 'label':...}）做批量生成并计算多分类指标。
    返回 dict 含 accuracy, macro precision/recall/f1, per-class metrics。
    """
    model.eval()
    preds = []
    truths = []

    batch_size = cfg["eval_batch_size"]
    max_new_tokens = cfg["gen_max_new_tokens"]
    max_prompt_length = cfg["max_length"]

    # 为避免重复 tokenize 过多次，我们先保存每 prompt 的 token 长度（用于切分生成序列）
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [make_prompt_and_target(it["text"], it["label"])[0] for it in batch]

        # 单独 token 以获得 prompt_len（逐条），并同时做 batch tokenization 用于生成
        prompt_encs = [tokenizer(p, truncation=True, max_length=max_prompt_length, add_special_tokens=True) for p in prompts]
        prompt_lens = [len(pe["input_ids"]) for pe in prompt_encs]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_length, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 对每个样本，根据 prompt_len 切分生成部分并解析
        for bi in range(len(batch)):
            pl = prompt_lens[bi]
            # 有时 generate 返回的序列长度与 input_ids padding 长度一致或更长
            seq = out_ids[bi].cpu().numpy().tolist()
            gen_part = seq[pl:] if pl < len(seq) else []
            gen_text = tokenizer.decode(gen_part, skip_special_tokens=True).strip()
            pred = parse_label_from_generated(gen_text)
            preds.append(pred)
            truths.append(batch[bi]["label"])

    model.train()
    # 计算指标
    acc = accuracy_score(truths, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(truths, preds, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(truths, preds, average='weighted', zero_division=0)
    per_class = precision_recall_fscore_support(truths, preds, average=None, labels=[0,1,2,3,4,5], zero_division=0)

    per_class_metrics = {}
    for idx, lab in enumerate([0,1,2,3,4,5]):
        per_class_metrics[lab] = {
            "precision": float(per_class[0][idx]),
            "recall": float(per_class[1][idx]),
            "f1": float(per_class[2][idx]),
        }

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class_metrics
    }
    return metrics


# ----------------------------
# 训练主函数
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_save_peft(model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {out_dir} (only adapter weights).")
    model.save_pretrained(out_dir)


def main(args):
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    device = cfg["device"]
    set_seed(cfg["seed"])

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # 一些 tokenizer 没有 pad_token，设为 eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 读数据
    train_items = read_tsv(str(Path(cfg["data_dir"]) / "train.txt"))
    val_items = read_tsv(str(Path(cfg["data_dir"]) / "val.txt"))
    test_items = read_tsv(str(Path(cfg["data_dir"]) / "test.txt"))

    train_ds = CLSDataset(train_items, tokenizer, cfg["max_length"])
    val_ds = CLSDataset(val_items, tokenizer, cfg["max_length"])
    test_ds = CLSDataset(test_items, tokenizer, cfg["max_length"])

    collate = functools.partial(collate_fn_with_pad, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["micro_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True
    )

    logger.info("Preparing BitsAndBytesConfig and loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    logger.info("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    logger.info("Applying LoRA...")
    model = get_peft_model(model, lora_config)

    # 打印可训练参数统计
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f"Trainable params: {trainable_params} / {all_params} ({100*trainable_params/all_params:.6f}%)")

    # 优化器 / scheduler
    optimizer_grouped_parameters = [
        {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": cfg["weight_decay"]}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg["lr"])

    total_update_steps = math.ceil(len(train_loader) * cfg["max_epochs"] / cfg["gradient_accumulation_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_update_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    global_step = 0
    best_val_f1 = 0.0
    model.train()
    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
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

            if (step + 1) % (cfg["gradient_accumulation_steps"] * 50) == 0:
                logger.info(f"Epoch {epoch} step {step+1}/{len(train_loader)} avg_loss={epoch_loss/(step+1):.6f}")

        t1 = time.time()
        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        logger.info(f"Epoch {epoch} finished in {(t1 - t0):.1f}s, avg_loss={avg_epoch_loss:.6f}")

        # 验证（生成方式）
        try:
            val_metrics = evaluate_generation(model, tokenizer, val_items, device, cfg)
            logger.info(f"Validation metrics after epoch {epoch}: {val_metrics}")
        except Exception as e:
            logger.warning(f"Validation generation failed: {e}")
            val_metrics = {}

        # 保存 adapter
        if cfg["save_every_epoch"]:
            out_dir = os.path.join(cfg["output_dir"], f"epoch{epoch}")
            safe_save_peft(model, out_dir)

        # Early save best f1 (macro)
        if "f1_macro" in val_metrics and val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            safe_save_peft(model, os.path.join(cfg["output_dir"], "best"))

    # 最终保存
    safe_save_peft(model, os.path.join(cfg["output_dir"], "final"))
    logger.info("Training complete.")


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
