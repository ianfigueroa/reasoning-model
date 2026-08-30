"""Plot GSM8K accuracy vs. thinking budget.

Reads every outputs/eval_*.json and plots accuracy against the number of forced
"Wait" injections (0 = no budget forcing). The upward curve is the test-time
scaling result.

    python plot_results.py
"""
import glob
import json

import matplotlib.pyplot as plt


def main():
    files = glob.glob("outputs/eval_*.json")
    if not files:
        print("no outputs/eval_*.json yet -- run eval_gsm8k.py first")
        return

    sft = []   # points from the SFT'd model, keyed by forced-thinking count
    base_acc = None
    for f in files:
        d = json.load(open(f))
        if d["adapter"]:
            sft.append((d["force"], d["accuracy"]))
        elif d["budget"] == "none":
            base_acc = d["accuracy"]

    sft.sort()
    if sft:
        xs, ys = zip(*sft)
        plt.plot(xs, ys, "o-", label="1.5B + SFT")
        for x, y in sft:
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8))
    if base_acc is not None:
        plt.axhline(base_acc, ls="--", color="gray", label=f"base ({base_acc:.2f})")

    plt.xlabel('forced "Wait" injections (thinking budget)')
    plt.ylabel("GSM8K accuracy")
    plt.title("Test-time scaling: accuracy vs. thinking budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/accuracy_vs_budget.png", dpi=150)
    print("wrote outputs/accuracy_vs_budget.png")


if __name__ == "__main__":
    main()
