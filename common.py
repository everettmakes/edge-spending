"""Shared pieces: categories, prompt format and scoring. Used in Colab and on the laptop."""
import json
import re

CATEGORIES = [
    "groceries", "transport", "eating_out", "bills",
    "subscriptions", "shopping", "entertainment", "cash",
]
EXPERIMENT = "edge-spending-categoriser"

INSTRUCTION = (
    "Categorise this bank transaction. Reply with JSON only, using the keys "
    '"merchant", "amount" and "category". '
    f"Category must be one of: {', '.join(CATEGORIES)}.\n"
    "Transaction: {line}"
)


def build_messages(line: str, answer: dict | None = None) -> list[dict]:
    """Chat messages for one transaction; include the assistant answer when training."""
    messages = [{"role": "user", "content": INSTRUCTION.format(line=line)}]
    if answer is not None:
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    return messages


def legacy_text(line: str, answer: dict) -> str:
    """The original tutorial format: different from the chat template used at inference, and no EOS."""
    return f"### Instruction:\n{INSTRUCTION.format(line=line)}\n\n### Response:\n{json.dumps(answer)}"


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


def score(predictions: list[str], labels: list[dict]) -> dict:
    """Aggregate metrics for a list of raw model replies against gold labels."""
    n = len(labels)
    valid = category = merchant = amount = exact = 0
    for reply, gold in zip(predictions, labels):
        pred = parse_json(reply)
        if pred is None:
            continue
        valid += 1
        cat_ok = str(pred.get("category", "")).strip().lower() == gold["category"]
        mer_ok = str(pred.get("merchant", "")).strip().lower() == gold["merchant"].lower()
        amt_ok = _amount_matches(pred.get("amount"), gold["amount"])
        category += cat_ok
        merchant += mer_ok
        amount += amt_ok
        exact += cat_ok and mer_ok and amt_ok
    return {
        "valid_json_rate": valid / n,
        "category_accuracy": category / n,
        "merchant_accuracy": merchant / n,
        "amount_accuracy": amount / n,
        "exact_match": exact / n,
    }


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]