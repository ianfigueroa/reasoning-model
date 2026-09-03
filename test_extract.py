"""CPU-only sanity check for the answer parsing. Runs before any ML install:

    python test_extract.py

If this passes, the scoring half of the eval is trustworthy -- a wrong parser
silently corrupts every accuracy number, so it's the one piece worth a test.
"""
from common import extract_answer, extract_gold, majority_vote


def test_boxed():
    # Math instruct models answer in \boxed{}; take the last box, parse the number.
    assert extract_answer(r"reasoning ... \boxed{72}") == 72
    assert extract_answer(r"step \boxed{18} then final \boxed{5}") == 5
    assert extract_answer(r"\boxed{1,024}") == 1024
    assert extract_answer(r"cost is \boxed{\$41}") == 41


def test_majority_vote():
    assert majority_vote([5, 5, 3, None, 5]) == 5   # ignores None, picks mode
    assert majority_vote([1, 2, 2, 1, 2]) == 2
    assert majority_vote([None, None]) is None
    assert majority_vote([7]) == 7


def test_gold():
    assert extract_gold("She sold 48+24 = 72 clips.\n#### 72") == 72
    assert extract_gold("...\n#### 1,024") == 1024
    assert extract_gold("no marker here") is None


def test_model_output():
    assert extract_answer("... so the total is 72.\n#### 72") == 72
    assert extract_answer("The answer is 41.") == 41          # cue, no ####
    assert extract_answer("steps: 10, 18, 28. Final: 35") == 35  # last number
    assert extract_answer("she has 2 apples and buys 3 more, #### 5") == 5
    assert extract_answer("3.5 dollars total. The answer is 3.5") == 3.5
    assert extract_answer("no digits at all") is None


def test_reasoning_number_not_grabbed():
    # A number buried in reasoning must not beat the explicit #### answer.
    assert extract_answer("first 100 then half is 50 ... #### 5") == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
