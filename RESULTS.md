# Results

Filling this in as I go. Same test subset and seed across every row so the numbers
are comparable.

Setup: base model `Qwen/Qwen2.5-1.5B-Instruct`, GSM8K test (first `N` questions),
greedy decoding, RTX 2070 8 GB, 4-bit QLoRA, fine-tuned on `simplescaling/s1K-1.1`.

## GSM8K accuracy

| Configuration              | Thinking budget | Accuracy |
|----------------------------|-----------------|----------|
| Base 1.5B (8-shot)         | —               | ___%     |
| Fine-tuned                 | none            | ___%     |
| Fine-tuned + budget        | force:2         | ___%     |
| Fine-tuned + budget        | force:4         | ___%     |
| Fine-tuned + budget        | force:8         | ___%     |

![accuracy vs thinking budget](outputs/accuracy_vs_budget.png)

## Notes

- Fine-tuning vs. base: _what changed._
- Thinking longer: _how far accuracy climbed and where it flattened out._
- Thinking less (`cap:N`): _the control, accuracy dropping as thinking gets cut._

## Resume line (fill X, Y once measured)

> Reproduced test-time scaling on a 1.5B model, raising GSM8K accuracy from X% to Y%
> with reasoning-trace fine-tuning (QLoRA) plus budget forcing, on a single GPU.
