#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate score predictions for ALL essays with prompt_id == 2
using Sakalti/Saka-14B with a LoRA adapter, and save them to JSONL.

Input JSONL should be produced by `taqeem_build_cot_jsonl.py` and contain keys:
  essay_id, prompt_id, trait, trait_name, prompt_text, essay,
  trait_prompt, cot_prompt, score_range

Example:
  python taqeem_prompt2_infer.py \

    --jsonl taqeem_train_CoT.jsonl \

    --model Sakalti/Saka-14B \

    --lora  ./saka_lora_cot \

    --out   predictions_prompt2.jsonl
"""
import argparse
import json
import re
from typing import Dict, Iterable, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TRAIT_ORDER = ["relevance", "organization", "vocabulary", "style", "development", "mechanics", "grammar"]

def load_entries(jsonl_path: str, prompt_id: int) -> List[Dict]:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if int(x.get("prompt_id", -1)) == int(prompt_id):
                entries.append(x)
    # Stable order: by essay_id then trait order
    entries.sort(key=lambda e: (e["essay_id"], TRAIT_ORDER.index(e["trait"]) if e["trait"] in TRAIT_ORDER else 999))
    return entries

def build_prompt(entry: Dict) -> str:
    parts = [
        f"سمة التقييم: {entry['trait_name']}",
        f"المجال: {entry['prompt_text']}",
        f"النص:\n{entry['essay']}",
        f"تعليمات التقييم:\n{entry['trait_prompt']}",
        f"سلسلة التفكير:\n{entry['cot_prompt']}",
        # Keep the output format consistent with training (score on first line, justification on next lines)
        f"الآن أعط درجة متوقعة من ({entry.get('score_range', '0-5')}) في السطر الأول، ثم برر القرار باختصار."
    ]
    return "\n\n".join(parts).strip()

_INT_RE = re.compile(r"(\d+)")

def extract_score(text: str, score_range: str) -> int:
    """Extract first integer and clamp to score_range if provided."""
    m = _INT_RE.search((text or "").strip())
    if not m:
        return None
    val = int(m.group(1))
    try:
        lo, hi = [int(z) for z in score_range.split("-")]
        val = max(lo, min(hi, val))
    except Exception:
        pass
    return val

def main(args: argparse.Namespace) -> None:
    # === Load model & adapter ===
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.lora)
    model.eval()

    # === Load data (prompt_id == 2) ===
    entries = load_entries(args.jsonl, args.prompt_id)
    print(f"Found {len(entries)} entries for prompt_id={args.prompt_id}.")

    written = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, entry in enumerate(entries, start=1):
            prompt = build_prompt(entry)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
                padding=False
            ).to(model.device)

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=False,  # deterministic for scoring
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                gen_out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Parse first line for numeric score
            first_line = generated.splitlines()[0] if generated else ""
            predicted_score = extract_score(first_line, entry.get("score_range", "0-5"))

            rec = {
                "essay_id": entry["essay_id"],
                "prompt_id": entry["prompt_id"],
                "trait": entry["trait"],
                "trait_name": entry["trait_name"],
                "predicted_score": predicted_score,
                "raw_output": generated,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            if i % args.log_every == 0:
                print(f"Processed {i}/{len(entries)} entries...")

    print(f"✅ Saved {written} predictions to {args.out}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer scores for prompt_id==2 and save as JSONL.")
    p.add_argument("--jsonl", type=str, default="taqeem_train_CoT.jsonl", help="Input JSONL path.")
    p.add_argument("--model", type=str, default="Sakalti/Saka-14B", help="Base model repo or path.")
    p.add_argument("--lora", type=str, default="./saka_lora_cot", help="LoRA adapter path.")
    p.add_argument("--out", type=str, default="predictions_prompt2.jsonl", help="Output JSONL path.")
    p.add_argument("--prompt_id", type=int, default=2, help="Prompt ID filter (default=2).")
    p.add_argument("--max_input_tokens", type=int, default=1024, help="Max tokens for input prompt.")
    p.add_argument("--max_new_tokens", type=int, default=80, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature.")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p (kept for completeness).")
    p.add_argument("--log_every", type=int, default=50, help="Progress log frequency.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)