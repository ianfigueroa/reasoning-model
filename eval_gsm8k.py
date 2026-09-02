"""GSM8K accuracy eval. One command per row of the results table:

    base model            -> python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200
    fine-tuned            -> python eval_gsm8k.py --adapter outputs/adapter --n 200
    fine-tuned + budget   -> python eval_gsm8k.py --adapter outputs/adapter --budget force:4 --n 200

Writes outputs/eval_<tag>.json so plot_results.py can draw the curve.
"""
import argparse
import json
import os
import time

from datasets import load_dataset

from common import build_prompt, extract_answer, extract_gold, load_model_4bit

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_budget(s):
    """'none' -> (0, None); 'force:N' -> (N, None); 'cap:N' -> (0, N)."""
    if not s or s == "none":
        return 0, None
    kind, _, n = s.partition(":")
    n = int(n)
    if kind == "force":
        return n, None
    if kind == "cap":
        return 0, n
    raise ValueError(f"bad --budget {s!r} (use none | force:N | cap:N)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (implies +SFT)")
    ap.add_argument("--n", type=int, default=200, help="how many test questions")
    ap.add_argument("--budget", default="none", help="none | force:N | cap:N")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    force, cap = parse_budget(args.budget)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    model, tok = load_model_4bit(args.model, adapter=args.adapter)

    # Only pull in the budget-forcing generate when we actually need it, so the
    # plain baseline path stays simple.
    if force or cap is not None:
        from budget_forcing import generate_with_budget
    else:
        import torch

    correct, rows = 0, []
    t0 = time.time()
    for i, ex in enumerate(ds):
        prompt = build_prompt(ex["question"])
        if force or cap is not None:
            text = generate_with_budget(model, tok, prompt, force=force, cap=cap,
                                        max_new_tokens=args.max_new_tokens)
        else:
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                # stop_strings halts the run-on: without it the model keeps
                # emitting fake "Question:/Answer:" pairs past the real answer.
                out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     stop_strings=["\nQuestion:", "\nQ:"], tokenizer=tok)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

        pred, gold = extract_answer(text), extract_gold(ex["answer"])
        ok = pred is not None and pred == gold
        correct += ok
        rows.append({"pred": pred, "gold": gold, "ok": ok})
        print(f"[{i+1}/{args.n}] pred={pred} gold={gold} {'OK' if ok else 'x'} "
              f"(running {correct}/{i+1})", flush=True)

    acc = correct / len(ds)
    tag = _tag(args)
    result = {"tag": tag, "model": args.model, "adapter": args.adapter,
              "budget": args.budget, "force": force, "n": len(ds),
              "accuracy": acc, "seconds": round(time.time() - t0, 1), "rows": rows}
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/eval_{tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\naccuracy = {acc:.3f} ({correct}/{len(ds)})  ->  {path}")


def _tag(args):
    base = "sft" if args.adapter else "base"
    b = args.budget.replace(":", "") if args.budget != "none" else "plain"
    return f"{base}_{b}"


if __name__ == "__main__":
    main()
