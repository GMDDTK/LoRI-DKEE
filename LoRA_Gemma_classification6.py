import os
import sys
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict
import functools
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit  # 使用 8bit 优化器

DEFAULTS = {
    "model_path": r"G:\pretrain_models\0Large Language Models\gemma-3-4b",
    "data_dir": r"E:\PythonProject\科学计量学谱系\知识元分类\classification6",
    "output_dir": "./lora_out_gemma_cls6",
    "micro_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_epochs": 3,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "warmup_steps": 50,
    "max_length": 256,  # 保持不变
    "seed": 42,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "save_every_epoch": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 六分类标签映射
LABEL_MAP = {
    0: "None/No classification (negative)",
    1: "Objective & Scope",
    2: "Data & Materials",
    3: "Methodology",
    4: "Results & Findings",
    5: "Discussion & Conclusion"
}

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def read_tsv(file_path: str) -> List[Dict]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or "\t" not in line:
                continue
            text, label = line.rsplit("\t", 1)
            if label.strip() not in {"0","1","2","3","4","5"}:
                continue
            items.append({"text": text.strip(), "label": int(label.strip())})
    return items

def make_prompt_and_target(text: str, label: int):
    mapping = "; ".join([f"{k}={v}" for k,v in LABEL_MAP.items()])
    prompt = (
        "Classify the following academic sentence into one of six categories. "
        "Reply with ONLY a single digit 0-5.\n"
        f"Categories: {mapping}\n\n"
        f"Sentence: {text}\n"
        "Answer (0-5):"
    )
    target = " " + str(label)
    return prompt, target

class CLSDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        prompt, target = make_prompt_and_target(it["text"], it["label"])
        full_text = prompt + target
        enc_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, add_special_tokens=True)
        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length, add_special_tokens=True)
        input_ids = enc_full["input_ids"]
        attention_mask = enc_full.get("attention_mask", [1] * len(input_ids))
        labels = [-100] * len(input_ids)
        for i in range(len(enc_prompt["input_ids"]), len(input_ids)):
            labels[i] = input_ids[i]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_token_id).long()
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": labels_padded}

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def safe_save_peft_model(model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {out_dir}")
    model.save_pretrained(out_dir)

def main(args):
    cfg = DEFAULTS.copy()
    cfg.update({k:v for k,v in vars(args).items() if v is not None})
    device = cfg["device"]
    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    train_items = read_tsv(str(Path(cfg["data_dir"]) / "train.txt"))
    if not train_items: sys.exit("No training data found.")
    train_ds = CLSDataset(train_items, tokenizer, cfg["max_length"])
    collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["micro_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=4,  # 多进程加载
        pin_memory=(device=="cuda")
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16,   # 用 bfloat16
        device_map="auto",
        trust_remote_code=True
    )
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 保持不变
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    optimizer = AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    total_update_steps = math.ceil(len(train_loader)/cfg["gradient_accumulation_steps"])*cfg["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_update_steps
    )

    model.train()
    global_step=0
    for epoch in range(1, cfg["max_epochs"]+1):
        epoch_loss=0.0
        t0=time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids=batch["input_ids"].to(device, non_blocking=True)
            attention_mask=batch["attention_mask"].to(device, non_blocking=True)
            labels=batch["labels"].to(device, non_blocking=True)

            outputs=model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss=outputs.loss/cfg["gradient_accumulation_steps"]
            loss.backward()

            if (step+1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step+=1

            epoch_loss += loss.item() * cfg["gradient_accumulation_steps"]

            # 打印训练进度
            if (step+1) % 20 == 0:
                avg_loss_so_far = epoch_loss / (step+1)
                logger.info(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                            f"Global Step {global_step} | Avg Loss {avg_loss_so_far:.6f}")

        t1=time.time()
        avg_loss=epoch_loss/len(train_loader)
        logger.info(f"Epoch {epoch} finished in {(t1-t0):.1f}s, avg_loss={avg_loss:.6f}")
        if cfg["save_every_epoch"]:
            out_dir=os.path.join(cfg["output_dir"], f"epoch_{epoch}")
            safe_save_peft_model(model, out_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--micro_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    args = parser.parse_args()
    main(args)
