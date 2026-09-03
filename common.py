"""Shared bits: GSM8K answer parsing, the CoT prompt, and 4-bit model loading.

Everything above load_model_4bit is torch-free on purpose so test_extract.py can
run (and CI can parse this) before the heavy ML stack is installed.
"""
import re

# The token the model uses to mark the end of its private thinking. Qwen chat
# models don't emit this by default, so we teach it during SFT and key budget
# forcing off it. Keep it in one place.
THINK_END = "</think>"

# 8-shot chain-of-thought exemplars (the classic GSM8K CoT prompt). Few-shot so
# the BASE model already reasons a little — otherwise the baseline is unfairly low
# and the SFT gain looks bigger than it really is.
FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?",
     "In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?",
     "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10.\n#### 10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half "
     "of the money she needs. Her parents decided to give her $15 for that purpose, "
     "and her grandparents twice as much as her parents. How much more money does "
     "Betty need to buy the wallet?",
     "Betty has 100 / 2 = $50. Her grandparents gave 15 * 2 = $30. Now she has "
     "50 + 15 + 30 = $95. She still needs 100 - 95 = $5.\n#### 5"),
    ("Julie is reading a 120-page book. Yesterday, she read 12 pages and today, she "
     "read twice as many pages as yesterday. If she wants to read half of the "
     "remaining pages tomorrow, how many pages should she read?",
     "Today she read 12 * 2 = 24 pages. So far she read 12 + 24 = 36 pages. Remaining "
     "= 120 - 36 = 84 pages. Half of that is 84 / 2 = 42.\n#### 42"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?",
     "Each time he writes 3 * 2 = 6 pages. Twice a week that's 6 * 2 = 12 pages. In a "
     "year that's 12 * 52 = 624 pages.\n#### 624"),
    ("Mark has a garden with flowers. He planted plants of three different colors in "
     "it. Ten of them are yellow, and there are 80% more of those in purple. There "
     "are only 25% as many green flowers as there are yellow and purple flowers. How "
     "many flowers does Mark have in his garden?",
     "Purple = 10 + 80% of 10 = 10 + 8 = 18. Yellow + purple = 10 + 18 = 28. Green = "
     "25% of 28 = 7. Total = 28 + 7 = 35.\n#### 35"),
    ("Alexis is applying for a new job and bought a new set of business clothes to "
     "wear to the interview. She went to a department store with a budget of $200 and "
     "spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on "
     "socks, and $18 on a belt. She also purchased a pair of shoes, but lost the "
     "receipt for them. She has $16 left from her budget. How much did Alexis pay for "
     "the shoes?",
     "She spent 30 + 46 + 38 + 11 + 18 = $143 on the known items. With $16 left she "
     "spent 200 - 16 = $184 total. So the shoes cost 184 - 143 = $41.\n#### 41"),
    ("Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is "
     "eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. "
     "If she works 10 hours every day for 5 days, how much money does she make?",
     "Overtime rate = 18 + 9 = $27. Each day: 8 * 18 = $144 regular plus 2 * 27 = $54 "
     "overtime = $198. Over 5 days that's 198 * 5 = $990.\n#### 990"),
]


# Chain-of-thought system prompt used by the Qwen2.5-Math instruct models. They
# are zero-shot with the chat template and put the final answer in \boxed{}.
BOXED_SYS = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(question):
    """8-shot CoT prompt ending on the target question."""
    parts = []
    for q, a in FEWSHOT:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def build_chat_prompt(question, tok):
    """Zero-shot chat-template prompt with the \\boxed{} CoT system prompt.

    This is how the Qwen instruct/math models are meant to be used; feeding them
    raw completion-style few-shot (build_prompt) leaves accuracy on the table.
    """
    msgs = [{"role": "system", "content": BOXED_SYS},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_gold(answer_field):
    """The gold answer from a GSM8K row is the number after '####'."""
    m = re.search(r"####\s*([-\d,\.]+)", answer_field)
    if not m:
        return None
    return _to_number(m.group(1))


def extract_answer(text):
    """Pull the model's final numeric answer.

    Prefer an explicit '#### N' (what we prompt for); otherwise fall back to the
    number after the first answer cue. Take the FIRST '####' — with greedy
    decoding and no stop token the model can run on and fabricate extra
    'Question/Answer' pairs, so the target question's answer is the first one,
    not the last.
    """
    # Math instruct models answer in \boxed{...}; take the LAST box (the final
    # answer, after any intermediate boxed steps) and pull the number out of it.
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", boxes[-1])
        if nums:
            return _to_number(nums[-1])
    hits = re.findall(r"####\s*([-\d,\.]+)", text)
    if hits:
        return _to_number(hits[0])
    tail = re.split(r"(?i)the answer is|answer:", text)[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    return _to_number(nums[-1])


def _to_number(s):
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def load_model_4bit(model_name, adapter=None):
    """Load a 4-bit (QLoRA) base model, optionally with a LoRA adapter on top.

    Imports live inside the function so this module stays importable without the
    ML stack (test_extract.py needs only the parsing above).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", dtype=torch.float16,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok
