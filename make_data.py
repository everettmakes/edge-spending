"""Generate synthetic, messy UK bank-statement lines with JSON labels.

Usage: python make_data.py --train 600 --test 150 --seed 42
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

# canonical merchant -> (category, raw statement variants, typical amount range in £)
MERCHANTS = {
    "Tesco":        ("groceries", ["TESCO STORES 3297", "TESCO EXPRESS", "TESCO.COM", "TESCO STORE"], (3, 120)),
    "Sainsbury's":  ("groceries", ["SAINSBURYS S/MKTS", "SAINSBURY'S LOCAL", "JS ONLINE GROCERY"], (3, 110)),
    "Aldi":         ("groceries", ["ALDI 71 BRISTOL", "ALDI STORES"], (5, 80)),
    "Lidl":         ("groceries", ["LIDL GB", "LIDL UK"], (4, 70)),
    "TfL":          ("transport", ["TFL TRAVEL CH", "TFL.GOV.UK/CP"], (2, 15)),
    "Trainline":    ("transport", ["TRAINLINE.COM", "TRAINLINE"], (8, 140)),
    "Uber":         ("transport", ["UBER *TRIP", "UBER* TRIP HELP.UBER.COM"], (6, 40)),
    "First Bus":    ("transport", ["FIRST BUS BRISTOL", "FIRSTGROUP BUS"], (2, 6)),
    "Nando's":      ("eating_out", ["NANDOS 0412", "NANDO'S CHICKENLAND"], (9, 45)),
    "Pret":         ("eating_out", ["PRET A MANGER", "PRET A MANGER (EUROPE)"], (3, 12)),
    "Deliveroo":    ("eating_out", ["DELIVEROO.CO.UK", "DELIVEROO"], (12, 45)),
    "Greggs":       ("eating_out", ["GREGGS PLC", "GREGGS 1823"], (2, 9)),
    "British Gas":  ("bills", ["BRITISH GAS", "BG SERVICES"], (40, 180)),
    "Thames Water": ("bills", ["THAMES WATER", "THAMES WATER UTIL"], (20, 70)),
    "Virgin Media": ("bills", ["VIRGIN MEDIA", "VIRGIN MEDIA PYMTS"], (25, 90)),
    "EE":           ("bills", ["EE LIMITED", "EE & T-MOBILE"], (10, 60)),
    "Netflix":      ("subscriptions", ["NETFLIX.COM", "NETFLIX INTERNATIONAL"], (5, 18)),
    "Spotify":      ("subscriptions", ["SPOTIFY UK", "SPOTIFY P1A2B3"], (6, 18)),
    "Amazon Prime": ("subscriptions", ["AMZNPRIME*AB12C", "AMAZON PRIME*UK"], (5, 95)),
    "Disney+":      ("subscriptions", ["DISNEY PLUS", "DISNEYPLUS"], (5, 11)),
    "Amazon":       ("shopping", ["AMZN MKTP UK*2H3", "AMAZON.CO.UK*AB1CD", "AMZ*AMAZON.CO.UK"], (4, 250)),
    "ASOS":         ("shopping", ["ASOS.COM LTD", "ASOS.COM"], (12, 150)),
    "Argos":        ("shopping", ["ARGOS LTD", "ARGOS RETAIL"], (8, 200)),
    "Boots":        ("shopping", ["BOOTS 1137", "BOOTS UK LTD"], (3, 60)),
    "Odeon":        ("entertainment", ["ODEON CINEMAS", "ODEON BRISTOL"], (8, 30)),
    "Steam":        ("entertainment", ["STEAM PURCHASE", "STEAMGAMES.COM 4259"], (3, 60)),
    "Ticketmaster": ("entertainment", ["TICKETMASTER UK", "TM *TICKETMASTER"], (20, 150)),
    "ATM":          ("cash", ["CASH WITHDRAWAL LNK", "ATM WDL NATWEST", "CASH BARCLAYS ATM"], (10, 200)),
}
CITIES = ["BRISTOL", "LONDON", "BATH", "CARDIFF", "MANCHESTER", ""]
TEMPLATES = [
    "{raw} {city} {date} £{amount}",
    "CARD PAYMENT TO {raw} ON {date} {amount}",
    "{raw} {ref} {amount} GBP",
    "CONTACTLESS {raw} {city} £{amount}",
    "DD {raw} REF {ref} £{amount}",
    "{date} {raw} {city} -{amount}",
]


def make_example(rng: random.Random) -> dict:
    merchant = rng.choice(list(MERCHANTS))
    category, raws, (low, high) = MERCHANTS[merchant]
    amount = round(rng.uniform(low, high), 2)
    amount_text = f"{amount:,.2f}" if rng.random() < 0.5 else f"{amount:.2f}"
    line = rng.choice(TEMPLATES).format(
        raw=rng.choice(raws),
        city=rng.choice(CITIES),
        date=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}",
        ref=f"{rng.randint(10000, 99999)}",
        amount=amount_text,
    )
    line = " ".join(line.split())  # collapse double spaces from empty cities
    return {"line": line, "label": {"merchant": merchant, "amount": amount, "category": category}}


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [make_example(rng) for _ in range(args.train + args.test)]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    write_jsonl(rows[: args.train], out / "train.jsonl")
    write_jsonl(rows[args.train:], out / "test.jsonl")

    counts = Counter(r["label"]["category"] for r in rows[: args.train])
    print(f"Wrote {args.train} train / {args.test} test rows to {out}/")
    print("Train categories:", dict(sorted(counts.items())))
    for row in rows[:3]:
        print(" ", row["line"], "->", row["label"])


if __name__ == "__main__":
    main()
