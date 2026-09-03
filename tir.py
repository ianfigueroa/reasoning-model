"""Tool-integrated reasoning (TIR) for the Qwen2.5-Math models.

The model writes Python, we actually run it and feed the real output back, and
it keeps going until it boxes an answer. This is the model card's higher-accuracy
mode -- it kills the mental-arithmetic slips that plain CoT makes.

Protocol (matches how Qwen2.5-Math is trained):
  system: "...integrate natural language reasoning with programs...\\boxed{}"
  the model emits a ```python ...``` block, we execute it, append a
  ```output ...``` block with the real result, and let it continue.

Security: the code comes from the model, so we run it in a subprocess with a
timeout and an AST allowlist/denylist first.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

TIR_SYS = ("Please integrate natural language reasoning with programs to solve "
           "the problem above, and put your final answer within \\boxed{}.")

# ponytail: allowlist imports + denylist dangerous names is the ceiling here. It
# stops the obvious os/subprocess/file/network escapes in model-written code;
# it is NOT a real jail (a payload hiding in an allowed lib could still bite).
# Fine for a trusted math model on your own machine -- upgrade to a container if
# this ever runs untrusted code.
_ALLOWED_IMPORTS = {"math", "cmath", "fractions", "decimal", "statistics",
                    "itertools", "functools", "collections", "re", "random",
                    "numpy", "sympy", "operator"}
_BANNED_NAMES = {"os", "sys", "subprocess", "socket", "shutil", "open", "exec",
                 "eval", "__import__", "compile", "input", "importlib",
                 "pathlib", "requests", "urllib", "globals", "locals"}

_CODE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def _is_safe(code):
    """Reject code that imports outside the math allowlist or touches the OS."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"import {a.name!r} not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                return False, f"import from {node.module!r} not allowed"
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return False, f"use of {node.id!r} not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder attribute access not allowed"
    return True, ""


def run_code(code, timeout=10):
    """Run model-written Python in a sandboxed subprocess, return stdout/stderr."""
    ok, why = _is_safe(code)
    if not ok:
        return f"[blocked: {why}]"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        return out[:2000] if out else "[no output]"
    except subprocess.TimeoutExpired:
        return "[timeout]"
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def generate_tir(model, tok, question, max_new_tokens=1024, max_rounds=3):
    """Run the write-code / execute / feed-back loop; return the full transcript."""
    import torch

    msgs = [{"role": "system", "content": TIR_SYS},
            {"role": "user", "content": question}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    transcript = ""
    for _ in range(max_rounds):
        ids = tok(prefix + transcript, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            # Stop as soon as the model starts an output block -- we substitute
            # the REAL execution result instead of its guessed one.
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 stop_strings=["```output"], tokenizer=tok)
        transcript += tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

        if "\\boxed{" in transcript:
            break
        blocks = list(_CODE.finditer(transcript))
        if not blocks:
            break  # no code and no answer -- nothing more to do
        # Drop anything after the last code block (e.g. a guessed ```output),
        # run the code for real, and hand the result back.
        last = blocks[-1]
        transcript = transcript[:last.end()]
        transcript += f"\n```output\n{run_code(last.group(1))}\n```\n"
    return transcript


def _demo():
    assert run_code("print(2 + 2)") == "4"
    assert run_code("import math; print(math.gcd(12, 18))") == "6"
    assert run_code("import os; os.system('echo hi')").startswith("[blocked")
    assert run_code("while True: pass") == "[timeout]"
    assert _CODE.search("x\n```python\nprint(1)\n```\n").group(1).strip() == "print(1)"
    print("tir self-check passed")


if __name__ == "__main__":
    _demo()
