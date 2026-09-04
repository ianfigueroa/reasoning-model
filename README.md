# Teaching a small model to reason

This takes a small open model (Qwen2.5-1.5B), fine-tunes it on about a thousand
step-by-step math solutions so it learns to "think" before answering, and then
measures how much its GSM8K accuracy goes up. It also plays with making the model
think longer or shorter at inference time and plots what that does to accuracy.

Everything runs on a single 8 GB GPU (RTX 2070) using 4-bit QLoRA. It's a small
reproduction of the s1 "simple test-time scaling" paper.

The point of the project is one number that goes up: GSM8K accuracy, base model vs.
after fine-tuning vs. after fine-tuning + budget forcing.

## What the ideas mean

**Test-time compute.** You can make a model better by training a bigger one, or by
letting an existing one think longer when it answers. The second one is cheap and
per-question. That's what o1 / DeepSeek-R1 do at scale; this is the tiny version.

**Chain-of-thought.** Instead of blurting the answer, the model writes its steps
("48 + 24 = 72, so..."). On math that alone helps a lot. A reasoning trace is just
one long worked-out solution like that.

**Distillation.** We don't teach reasoning from scratch, we copy it. A strong model
already wrote ~1,000 good traces (the s1K dataset) and we fine-tune ours to imitate
them. The surprise from the paper is that ~1k is enough.

**SFT.** Supervised fine-tuning: show the model the input and the ideal output and
have it copy the pattern. Here it learns to emit a `<think>...</think>` block before
the answer.

**LoRA.** Instead of retraining all 1.5B weights, freeze them and train a few small
adapter matrices (under 1% of the params). You only save the little adapter.

**QLoRA.** Same thing but the frozen base model is loaded in 4-bit, which is what
lets a 1.5B model fit and train in 8 GB.

**Budget forcing.** The inference-time lever, no retraining involved:
- think longer (`force:N`) - when the model tries to close its thinking with
  `</think>`, drop that token and stick "Wait" on the end so it keeps going, up to N
  times. More thinking usually means more accuracy, up to a point.
- think less (`cap:N`) - just cut the thinking off after N tokens. The control.

**GSM8K.** ~8k grade-school math word problems with a single numeric answer. The
usual benchmark for whether a small model can actually reason.

One thing I was careful about: measure the base model *first*, before any training,
on a fixed set of questions with a fixed prompt and greedy decoding, then reuse the
exact same setup for every later run. Otherwise the "improvement" could just be the
measurement moving around.

## Running it

Get the CUDA build of torch, not the CPU one. Python 3.13 is fine, but the cp313
torch wheels live on the cu126 index, not the old cu121 one. bitsandbytes has a
proper Windows wheel now so QLoRA works natively; if the CUDA side ever acts up, the
same code runs in WSL2.

```bash
cd ~/reasoning-model
python -m venv .venv
.venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# sanity-check the scorer, no GPU needed
python test_extract.py

# baseline first
python eval_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200

# fine-tune (a couple hours on the 2070)
python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --max_samples 1000

# re-eval with the adapter, then push the thinking budget up
python eval_gsm8k.py --adapter outputs/adapter --n 200
python eval_gsm8k.py --adapter outputs/adapter --budget force:2 --n 200
python eval_gsm8k.py --adapter outputs/adapter --budget force:4 --n 200
python eval_gsm8k.py --adapter outputs/adapter --budget force:8 --n 200

# draw the accuracy-vs-budget curve
python plot_results.py
```

Each eval drops an `outputs/eval_<tag>.json`, and `plot_results.py` turns those into
`outputs/accuracy_vs_budget.png`. I keep the actual numbers in RESULTS.md.

## Files

| File | What it does |
|------|------------|
| `common.py` | answer parsing, the 8-shot prompt, 4-bit model loading |
| `eval_gsm8k.py` | scores GSM8K accuracy (base, fine-tuned, or budget-forced) |
| `train_sft.py` | QLoRA fine-tune on the s1K traces |
| `budget_forcing.py` | the think-longer / think-less generation |
| `test_extract.py` | quick test for the answer parser |
| `plot_results.py` | the accuracy-vs-budget plot |
| `RESULTS.md` | the numbers |

## Scope

The goal is a clean, honest gain (something like +5 to +15 points) with a writeup
anyone can reproduce, not a state-of-the-art number. Test-time compute can't make up
for knowledge a model just doesn't have, which is why this stays at 1.5B and not
smaller. RL-based reasoning (GRPO/PPO) would be the obvious next step but I left it
out of the first version.

Based on the s1 test-time scaling work by Muennighoff et al.
