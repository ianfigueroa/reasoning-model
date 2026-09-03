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

from common import (build_chat_prompt, build_prompt, extract_answer,
                    extract_gold, load_model_4bit, majority_vote)

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
    ap.add_argument("--prompt", choices=["fewshot", "chat", "tir"], default="fewshot",
                    help="fewshot = 8-shot completion (base repro); "
                         "chat = zero-shot chat template + \\boxed{} (instruct/math models); "
                         "tir = tool-integrated reasoning, model runs Python (math models)")
    ap.add_argument("--samples", type=int, default=1,
                    help="self-consistency: sample N paths and majority-vote (1 = greedy)")
    ap.add_argument("--tir_rounds", type=int, default=3,
                    help="max code-execution rounds for --prompt tir")
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
    if args.prompt == "tir":
        from tir import generate_tir

    correct, rows = 0, []
    t0 = time.time()
    for i, ex in enumerate(ds):
        if args.prompt == "tir":
            # model writes + runs Python; texts stays a 1-list so scoring is uniform
            texts = [generate_tir(model, tok, ex["question"],
                                  max_new_tokens=args.max_new_tokens,
                                  max_rounds=args.tir_rounds)]
        else:
            prompt = (build_chat_prompt(ex["question"], tok) if args.prompt == "chat"
                      else build_prompt(ex["question"]))
            if force or cap is not None:
                texts = [generate_with_budget(model, tok, prompt, force=force, cap=cap,
                                              max_new_tokens=args.max_new_tokens)]
            else:
                ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
                with torch.no_grad():
                    # stop_strings halts the run-on: without it the model keeps
                    # emitting fake "Question:/Answer:" pairs past the real answer.
                    gen = dict(max_new_tokens=args.max_new_tokens,
                               pad_token_id=tok.pad_token_id,
                               stop_strings=["\nQuestion:", "\nQ:"], tokenizer=tok)
                    if args.samples > 1:
                        # self-consistency: sample N paths in one batched call
                        gen.update(do_sample=True, temperature=0.7, top_p=0.9,
                                   num_return_sequences=args.samples)
                    else:
                        gen.update(do_sample=False)
                    out = model.generate(ids, **gen)
                texts = [tok.decode(out[j, ids.shape[1]:], skip_special_tokens=True)
                         for j in range(out.shape[0])]

        text = texts[0]
        gold = extract_gold(ex["answer"])
        pred = majority_vote([extract_answer(t) for t in texts])
        ok = pred is not None and pred == gold
        correct += ok
        rows.append({"pred": pred, "gold": gold, "ok": ok})
        print(f"[{i+1}/{args.n}] pred={pred} gold={gold} {'OK' if ok else 'x'} "
              f"(running {correct}/{i+1})", flush=True)

    acc = correct / len(ds)
    tag = _tag(args)
    result = {"tag": tag, "model": args.model, "adapter": args.adapter,
              "budget": args.budget, "force": force, "samples": args.samples,
              "prompt": args.prompt, "n": len(ds),
              "accuracy": acc, "seconds": round(time.time() - t0, 1), "rows": rows}
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/eval_{tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\naccuracy = {acc:.3f} ({correct}/{len(ds)})  ->  {path}")


def _tag(args):
    base = "sft" if args.adapter else "base"
    b = args.budget.replace(":", "") if args.budget != "none" else "plain"
    # Short model slug + prompt mode so different models/modes don't clobber each
    # other's json (e.g. base_plain vs math_chat).
    slug = args.model.rsplit("/", 1)[-1].lower()
    if "math" in slug:
        base = "math"
    tag = f"{base}_{b}"
    if args.prompt in ("chat", "tir"):
        tag += f"_{args.prompt}"
    if args.samples > 1:
        tag += f"_sc{args.samples}"
    return tag


if __name__ == "__main__":
    main()
