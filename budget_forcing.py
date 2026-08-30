"""Budget forcing, the test-time-compute lever from the s1 paper.

Two knobs on how long the model gets to think before it answers:

  - think longer (force N): every time the model tries to end its thinking
    (emits </think>), swallow that token and glue "Wait" onto the transcript so
    it keeps reasoning. Do this up to `force` times. More thinking tends to raise
    accuracy, up to a point.

  - think less (cap N): stop the thinking after N tokens and make it answer. A
    control that shows the other end of the curve.

Both happen at inference, no retraining. That's the whole idea of test-time
compute: spend more at answer time instead of train time.
"""
import torch

from common import THINK_END


def generate_with_budget(model, tok, prompt, force=0, cap=None,
                         max_new_tokens=1024, answer_tokens=256):
    """Generate a completion under a thinking budget.

    force: number of times to suppress </think> and inject "Wait" (think longer).
    cap:   hard limit on thinking tokens before forcing the answer (think less).
    Returns the decoded completion (prompt stripped).
    """
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    start_len = ids.shape[1]
    think_end_ids = tok(THINK_END, add_special_tokens=False).input_ids
    wait_ids = tok(" Wait", add_special_tokens=False).input_ids

    injections = 0
    thinking_budget = cap if cap is not None else max_new_tokens

    # Phase 1: think, honoring cap and the force-more injections.
    while ids.shape[1] - start_len < thinking_budget:
        step = _gen(model, ids, max_new_tokens=64)
        ids = torch.cat([ids, step], dim=1)
        if _ends_with(ids, think_end_ids):
            if injections < force:
                # swallow the </think> we just emitted, append "Wait", keep going
                ids = ids[:, :-len(think_end_ids)]
                ids = torch.cat([ids, _t(wait_ids, device)], dim=1)
                injections += 1
                continue
            break  # model is done thinking and we're not forcing more
        if _hit_eos(step, tok):
            break

    # Phase 2: make sure it actually answers after thinking stops.
    if not _ends_with(ids, tok(tok.eos_token, add_special_tokens=False).input_ids):
        ids = torch.cat([ids, _t(think_end_ids, device)], dim=1)
        step = _gen(model, ids, max_new_tokens=answer_tokens)
        ids = torch.cat([ids, step], dim=1)

    return tok.decode(ids[0, start_len:], skip_special_tokens=True)


def _gen(model, ids, max_new_tokens):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=model.config.eos_token_id or 0)
    return out[:, ids.shape[1]:]


def _ends_with(ids, suffix):
    n = len(suffix)
    return n > 0 and ids.shape[1] >= n and ids[0, -n:].tolist() == suffix


def _hit_eos(step, tok):
    return tok.eos_token_id in step[0].tolist()


def _t(id_list, device):
    return torch.tensor([id_list], device=device)
