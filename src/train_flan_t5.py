#!/usr/bin/env python3
"""
Fine-tune FLAN-T5 on paper-level JSONL dataset.

Expected files:
  data/processed/train.jsonl
  data/processed/val.jsonl

Each JSONL row must include:
  - input_text
  - target_text

Example run:
  python3 src/train_flan_t5.py --config configs/run1_reviews_only.yaml
or:
  python3 src/train_flan_t5.py --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# LoRA / PEFT
from peft import LoraConfig, get_peft_model, TaskType


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    model_name: str = "google/flan-t5-base"
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
    output_dir: str = "runs/flan_t5_run1"

    # tokenization
    max_input_length: int = 1024
    max_target_length: int = 512

    # training
    seed: int = 42
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2

    fp16: bool = False  # MPS doesn't use fp16 the same way; keep False on Mac
    bf16: bool = False

    # generation during eval
    predict_with_generate: bool = True
    generation_max_new_tokens: int = 256
    generation_num_beams: int = 4

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def dict_to_config(d: Dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config.")
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg_dict = load_yaml(args.config)
        cfg = dict_to_config(cfg_dict)
    else:
        cfg = Config()

    # CLI overrides
    if args.train_path:
        cfg.train_path = args.train_path
    if args.val_path:
        cfg.val_path = args.val_path
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config used for this run
    with open(Path(cfg.output_dir) / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    device = get_device()
    print(f"✅ Using device: {device}")

    set_seed(cfg.seed)

    # Load dataset (JSONL)
    data_files = {"train": cfg.train_path, "validation": cfg.val_path}
    ds = load_dataset("json", data_files=data_files)
    print(ds)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # LoRA (recommended for Mac)
    if cfg.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print("✅ LoRA enabled. Trainable params:")
        model.print_trainable_parameters()

    # Tokenization
    def preprocess(batch):
        inputs = batch["input_text"]
        targets = batch["target_text"]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_input_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=cfg.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
        num_proc=1,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    # Training args (MPS-safe)
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        logging_steps=cfg.logging_steps,
        report_to="none",

        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.max_input_length + cfg.generation_max_new_tokens,
        generation_num_beams=cfg.generation_num_beams,

        # On Apple Silicon, keep fp16/bf16 off
        fp16=cfg.fp16,
        bf16=cfg.bf16,

        # Speeds up training a bit
        torch_compile=False,
        dataloader_pin_memory=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)         # saves adapter when using PEFT
    tokenizer.save_pretrained(cfg.output_dir)
    model.config.save_pretrained(cfg.output_dir)

    # Save metrics
    metrics = train_result.metrics
    with open(Path(cfg.output_dir) / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics.items()}, f, indent=2)

    print("✅ Training complete. Model saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()