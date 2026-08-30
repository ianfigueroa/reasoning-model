"""Fine-tune the reasoning behavior with QLoRA.

QLoRA is a 4-bit frozen base model plus small trainable LoRA adapters, which is
what makes a 1.5B model fine-tunable in 8 GB. We train on ~1k long reasoning
traces (s1K) so the model learns to write a <think>...</think> block before its
answer, which is the behavior budget forcing later leans on.

    python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --max_samples 1000
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import SFTConfig, SFTTrainer

from common import THINK_END


def format_example(ex):
    """s1K rows carry a question, a long reasoning trace, and a solution. Wrap the
    trace in <think>...</think> so we're explicitly training the thinking span.
    Prefer the DeepSeek trace (longer/cleaner), fall back to the Gemini one."""
    q = ex.get("question", "")
    think = (ex.get("deepseek_thinking_trajectory")
             or ex.get("gemini_thinking_trajectory") or "")
    ans = (ex.get("solution")
           or ex.get("deepseek_attempt") or ex.get("gemini_attempt") or "")
    text = f"Question: {q}\nAnswer: <think>{think}{THINK_END}{ans}"
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="simplescaling/s1K-1.1")
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="outputs/adapter")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(format_example, remove_columns=ds.column_names)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", dtype=torch.float16)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        max_length=args.max_seq_len, logging_steps=10, save_strategy="epoch",
        bf16=False, fp16=True, gradient_checkpointing=True,
        optim="paged_adamw_8bit", report_to="none", dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
