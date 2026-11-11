"""Shared pieces: categories, prompt format and scoring. Used in Colab and on the laptop."""
import json
import re

CATEGORIES = [
    "groceries", "transport", "eating_out", "bills",
    "subscriptions", "shopping", "entertainment", "cash",
]
EXPERIMENT = "edge-spending-categoriser-v3"  # v3 = fixed test sets + recipe ablation + size ladder (v1, v2 kept)

INSTRUCTION = (
    "Categorise this bank transaction. Reply with JSON only, using the keys "
    '"merchant", "amount" and "category". '
    "Category must be one of: {categories}.\n"
    "Transaction: {line}"
)


def prompt(line: str, categories: list[str] | None = None) -> str:
    """The category list is part of the prompt, so a user-defined category can be added later.
    Training shuffles its order (prepare_mlx_data.py) so the model learns to read the list."""
    return INSTRUCTION.format(categories=", ".join(categories or CATEGORIES), line=line)


def build_messages(line: str, answer: dict | None = None, categories: list[str] | None = None) -> list[dict]:
    """Chat messages for one transaction; include the assistant answer when training."""
    messages = [{"role": "user", "content": prompt(line, categories)}]
    if answer is not None:
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    return messages


def legacy_text(line: str, answer: dict, categories: list[str] | None = None) -> str:
    """The original tutorial format: different from the chat template used at inference."""
    return f"### Instruction:\n{prompt(line, categories)}\n\n### Response:\n{json.dumps(answer)}"


def parse_json(text: str) -> dict | None:
    """Pull the first {...} object out of a model reply; None if there isn't a valid one."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _amount_matches(predicted, expected: float) -> bool:
    try:
        return abs(float(str(predicted).replace(",", "").replace("£", "")) - expected) < 0.005
    except (TypeError, ValueError):
        return False


def norm_merchant(name) -> str:
    """Compare merchant names ignoring case, spaces and punctuation: Sainsburys == Sainsbury's, E.ON == EON."""
    return re.sub(r"[^a-z0-9+&]", "", str(name).lower())


def score_one(reply: str, gold: dict) -> dict:
    """Per-line correctness. results.py uses these to run paired significance tests between models."""
    pred = parse_json(reply)
    if pred is None:
        return {"valid": False, "category": False, "merchant": False, "amount": False, "exact": False}
    cat_ok = str(pred.get("category", "")).strip().lower() == gold["category"]
    mer_ok = norm_merchant(pred.get("merchant", "")) == norm_merchant(gold["merchant"])
    amt_ok = _amount_matches(pred.get("amount"), gold["amount"])
    return {"valid": True, "category": cat_ok, "merchant": mer_ok, "amount": amt_ok,
            "exact": cat_ok and mer_ok and amt_ok}


def score(predictions: list[str], labels: list[dict]) -> dict:
    """Aggregate metrics for a list of raw model replies against gold labels."""
    per_line = [score_one(reply, gold) for reply, gold in zip(predictions, labels)]
    n = len(per_line)
    mean = lambda key: sum(row[key] for row in per_line) / n  # noqa: E731
    return {
        "valid_json_rate": mean("valid"),
        "category_accuracy": mean("category"),
        "merchant_accuracy": mean("merchant"),
        "amount_accuracy": mean("amount"),
        "exact_match": mean("exact"),
    }


# Test splits: "seen" uses training merchants; "unseen" uses held-out merchants and line formats;
# "real" is your own hand-labelled bank lines (optional, skipped if the file doesn't exist).
# Metrics are logged with a "<split>_" prefix, except seen which keeps the plain names.
SPLITS = {"seen": "data/test.jsonl", "unseen": "data/test_unseen.jsonl", "real": "data/test_real.jsonl"}


def metric_prefix(split: str) -> str:
    return "" if split == "seen" else f"{split}_"


# Base-model ladder for the size experiment. Any Hugging Face / MLX repo id also works.
MODELS = {
    "qwen-3b-4bit": "mlx-community/Qwen2.5-3B-Instruct-4bit",       # size ladder, all QLoRA on a 4-bit base
    "qwen-1.5b-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen-0.5b-4bit": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",                       # 494M, bf16
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",           # 362M, bf16
    "gemma3-270m": "mlx-community/gemma-3-270m-it-bf16",             # 268M, bf16
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",           # 135M, bf16
}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
