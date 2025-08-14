#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning for Sakalti/Saka-14B on TAQEEM CoT JSONL.

Expects input JSONL created by taqeem_build_cot_jsonl.py.

Example:
  python taqeem_saka14b_lora_train.py \

    --jsonl taqeem_train_CoT.jsonl \

    --prompt-id 1 \

    --model Sakalti/Saka-14B \

    --out ./saka_lora_cot
"""
import argparse
import json
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

def load_and_filter(jsonl_path: str, prompt_id: int) -> List[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if prompt_id is not None:
        data = [x for x in data if int(x["prompt_id"]) == int(prompt_id)]
    print(f"✅ Samples after filtering (prompt_id={prompt_id}): {len(data)}")
    return data

def build_prompt_and_completion(entry: Dict) -> Dict[str, str]:
    input_parts = [
        f"سمة التقييم: {entry['trait_name']}",
        f"المجال: {entry['prompt_text']}",
        f"النص:\n{entry['essay']}",
        f"تعليمات التقييم:\n{entry['trait_prompt']}",
        f"سلسلة التفكير:\n{entry['cot_prompt']}",
        f"{entry['show_gold']}",
    ]
    prompt = "\n\n".join(input_parts).strip()
    completion = f"{entry['gold_score']}\n{entry['justification']}"
    return {"prompt": prompt, "completion": completion}

def main(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("medium")

    # === Data ===
    raw = load_and_filter(args.jsonl, args.prompt_id)
    pairs = [build_prompt_and_completion(x) for x in raw]

    # === Tokenizer & Model ===
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Choose dtype
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # === LoRA ===
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === Preprocess ===
    max_length = args.max_length

    def preprocess(batch):
        prompts = batch["prompt"]
        completions = batch["completion"]
        input_texts = [p + "\n" + c for p, c in zip(prompts, completions)]
        tokenized = tokenizer(
            input_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = []
        for p, ids in zip(prompts, tokenized["input_ids"]):
            prompt_len = len(tokenizer(p + "\n")["input_ids"])
            label = ids[:]
            label[:prompt_len] = [-100] * prompt_len  # Only compute loss on completion
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    ds = Dataset.from_list(pairs).map(preprocess, batched=True, remove_columns=["prompt", "completion"])

    # === Collator ===
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Training ===
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # === Save ===
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"✅ LoRA adapter and tokenizer saved to {args.out}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Saka-14B on TAQEEM CoT.")
    p.add_argument("--jsonl", type=str, default="taqeem_train_CoT.jsonl", help="Input JSONL path.")
    p.add_argument("--prompt-id", type=int, default=1, help="Filter by prompt_id (use -1 to disable).")
    p.add_argument("--model", type=str, default="Sakalti/Saka-14B", help="Base model repo or path.")
    p.add_argument("--out", type=str, default="./saka_lora_cot", help="Output directory.")
    p.add_argument("--max_length", type=int, default=1024, help="Max sequence length.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size.")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    p.add_argument("--lora_dropout", type=float, default=0.08, help="LoRA dropout.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.prompt_id == -1:
        # Keep all prompt_ids
        pass
    main(args)