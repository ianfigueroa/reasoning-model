# Results

Measured on 2026-09-03. Same test subset (first N=100 GSM8K test questions)
across every row, so the numbers are comparable. Hardware: RTX 2070 8 GB, 4-bit
(QLoRA nf4) inference.

## GSM8K accuracy

| Configuration | Prompt / decode | Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 8-shot CoT, greedy | **53%** |
| + s1 QLoRA SFT (`s1K-1.1`) | greedy | 52% |
| + s1 SFT, budget force:2 | greedy | 54% |
| + s1 SFT, budget force:4 | greedy | 54% |
| + s1 SFT, budget force:8 | greedy | 54% |
| **Qwen2.5-Math-1.5B-Instruct** | zero-shot chat + `\boxed{}`, greedy | **87%** |
| Qwen2.5-Math-1.5B-Instruct | + self-consistency (5 paths, vote) | 87% |
| Qwen2.5-Math-1.5B-Instruct | + TIR (model writes & runs Python) | 83% |

![accuracy vs thinking budget](outputs/accuracy_vs_budget.png)

## Notes

- **Eval-harness bug (fixed).** An earlier version of this eval reported base
  accuracy of 13%. That was a measurement artifact, not the model: greedy decode
  with **no stop sequence** let the 1.5B model run on emitting fake
  `Question:/Answer:` pairs past its real answer, and the parser then took the
  **last** `####` (a fabricated one). Fixing the stop sequence + taking the first
  `####` moved the *same* base model from 13% to its true ~53% — right where
  Qwen2.5-1.5B-Instruct is expected to score. Lesson: always sanity-check a
  baseline against the published number before trusting a "gain."

- **s1 SFT / budget forcing: no gain at 1.5B.** Reasoning-trace SFT on
  `s1K-1.1` and s1-style budget forcing held flat at 52–54% — statistically
  indistinguishable from the 53% base (n=100, SE ≈ ±5%). The s1 "more
  test-time compute → higher accuracy" effect was demonstrated on a 32B model
  and **does not reproduce at 1.5B scale**: the small model can't exploit the
  extra thinking. This is a genuine negative result, not a bug.

- **Biggest lever: the right base model + correct prompting (+34 pts).**
  Swapping to the same-size math-specialized `Qwen2.5-Math-1.5B-Instruct` and
  prompting it the way it was trained (chat template + "reason step by step,
  put your final answer in `\boxed{}`", zero-shot) jumped accuracy 53% → **87%**
  with no extra training and the same 8 GB GPU.

- **Self-consistency confirms, doesn't add.** Sampling 5 paths and majority-
  voting landed on the exact same 87%. Majority-vote helps most when the model
  is *uncertain*; a confident math model has little to gain. The upside: two
  independent decoding methods agree, so 87% is robust.

- **TIR (tool-integrated reasoning) actually hurt a little: 83%.** Letting the
  model write and run real Python (sandboxed subprocess, `tir.py`) was supposed
  to kill arithmetic slips. It works mechanically — 0 of the 17 misses were
  parse failures, all were genuine reasoning errors — but it lands *below* plain
  chat CoT. The reason: GSM8K misses at this model size are mostly *conceptual*
  (misreading the problem, e.g. setting up "increased by 150%" wrong), and TIR
  fixes arithmetic, not comprehension. The published ~95% TIR numbers are for the
  7B/72B math models; the effect doesn't carry down to 1.5B. Another honest
  negative result, same theme as budget forcing.

- **The pattern across this whole study:** at 1.5B scale, the fancy test-time
  methods (s1 budget forcing, self-consistency, TIR) don't move the needle — the
  only thing that did was picking the right base model and prompting it correctly.
  The realistic path to ~95% is a bigger base model (Qwen2.5-Math-7B in 4-bit),
  not a cleverer decode.

## Resume line

> Diagnosed and fixed a silent eval-harness bug that had understated a GSM8K
> baseline by 40 points (missing stop sequence + wrong answer-span parse), then
> ran a clean ablation on a single 8 GB GPU: 4-bit QLoRA SFT, s1-style test-time
> budget forcing (reproduced as a negative result at 1.5B), and self-consistency
> — reaching **87% GSM8K** by moving to a math-specialized 1.5B model with
> correct chat-template prompting.
