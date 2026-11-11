"""Generate synthetic, messy UK bank-statement lines with JSON labels, split by MERCHANT.

Every merchant belongs to exactly one split, so validation and the unseen test measure generalisation
to merchants the model has never seen, not memorisation.

  data/train.jsonl        train merchants, train line formats
  data/valid.jsonl        held-out merchants, train line formats   -> val_loss + checkpoint selection
  data/test.jsonl         "seen": train merchants, new lines         -> memorisation check
  data/test_unseen.jsonl  "unseen": test merchants AND held-out line formats -> the headline number
  data/splits.json        which merchant went where

v3 recipe additions (both switchable, for the ablation):
  --no-extra     drop the ~70 extra merchants, which only ever go to train/valid
  --no-augment   drop random bank-code prefixes and store-number/suffix noise on train/valid lines
Test files use their own random streams and the original merchant split, so they are identical whatever
the recipe flags: recipes can be compared fairly on the same test lines.

Usage: python make_data.py                                  # v3 recipe
       python make_data.py --out data_v2recipe --no-augment --no-extra   # ablation: same tests
"""
#----------IMPORTS----------#
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

#----------MERCHANTS----------#

# (canonical merchant, category, raw statement variants, typical amount range in £)
MERCHANTS = [
    # groceries
    ("Tesco", "groceries", ["TESCO STORES 3297", "TESCO EXPRESS", "TESCO.COM"], (3, 120)),
    ("Sainsbury's", "groceries", ["SAINSBURYS S/MKTS", "SAINSBURY'S LOCAL", "JS ONLINE GROCERY"], (3, 110)),
    ("Asda", "groceries", ["ASDA STORES", "ASDA SUPERSTORE 4412", "ASDA.COM"], (4, 120)),
    ("Aldi", "groceries", ["ALDI 71 BRISTOL", "ALDI STORES"], (5, 80)),
    ("Lidl", "groceries", ["LIDL GB", "LIDL UK"], (4, 70)),
    ("Waitrose", "groceries", ["WAITROSE 0435", "WAITROSE & PARTNERS"], (5, 110)),
    ("Morrisons", "groceries", ["WM MORRISONS STORE", "MORRISONS 0218"], (4, 90)),
    ("Co-op", "groceries", ["CO-OP GROUP FOOD", "CO OP FOOD 0291"], (2, 40)),
    ("Iceland", "groceries", ["ICELAND FOODS", "ICELAND 0773"], (4, 60)),
    ("M&S", "groceries", ["M&S SIMPLY FOOD", "MARKS&SPENCER FOOD"], (3, 50)),
    ("Ocado", "groceries", ["OCADO RETAIL", "OCADO.COM"], (40, 150)),
    ("Spar", "groceries", ["SPAR 2231", "SPAR STORES"], (2, 25)),
    ("Farmfoods", "groceries", ["FARMFOODS LTD", "FARMFOODS 0114"], (5, 50)),
    ("Booths", "groceries", ["BOOTHS SUPERMARKET", "BOOTHS 0021"], (5, 80)),
    ("Costcutter", "groceries", ["COSTCUTTER 118", "COSTCUTTER SUPERMKT"], (2, 25)),
    ("Budgens", "groceries", ["BUDGENS 0332", "BUDGENS STORES"], (2, 40)),
    ("Whole Foods", "groceries", ["WHOLE FOODS MARKET", "WHOLEFDS KEN 1032"], (5, 70)),
    ("Planet Organic", "groceries", ["PLANET ORGANIC", "PLANET ORGANIC LTD"], (4, 50)),
    # transport
    ("TfL", "transport", ["TFL TRAVEL CH", "TFL.GOV.UK/CP"], (2, 15)),
    ("Trainline", "transport", ["TRAINLINE.COM", "TRAINLINE"], (8, 140)),
    ("Uber", "transport", ["UBER *TRIP", "UBER* TRIP HELP.UBER.COM"], (6, 40)),
    ("First Bus", "transport", ["FIRST BUS BRISTOL", "FIRSTGROUP BUS"], (2, 6)),
    ("National Express", "transport", ["NATIONAL EXPRESS", "NATEXPRESS.COM"], (8, 60)),
    ("GWR", "transport", ["GWR TICKET OFFICE", "GREAT WESTERN RAILWAY"], (5, 120)),
    ("Avanti West Coast", "transport", ["AVANTI WEST COAST", "AVANTIWESTCOAST.CO.UK"], (15, 150)),
    ("LNER", "transport", ["LNER.CO.UK", "LNER TICKETS"], (15, 150)),
    ("Stagecoach", "transport", ["STAGECOACH BUS", "STAGECOACH SOUTH"], (2, 8)),
    ("Bolt", "transport", ["BOLT.EU/O/2401", "BOLT RIDE"], (5, 35)),
    ("Lime", "transport", ["LIME*RIDE", "LIME RIDE UK"], (1, 10)),
    ("Shell", "transport", ["SHELL 4471", "SHELL UK OIL"], (20, 90)),
    ("BP", "transport", ["BP 7731", "BP CONNECT"], (20, 90)),
    ("Esso", "transport", ["ESSO 3321", "ESSO TEXACO"], (20, 90)),
    ("RingGo", "transport", ["RINGGO PARKING", "RINGGO*BRISTOL"], (1, 12)),
    ("easyJet", "transport", ["EASYJET.COM", "EASYJET AIRLINE"], (25, 200)),
    ("Ryanair", "transport", ["RYANAIR", "RYANAIR LTD"], (20, 180)),
    ("Megabus", "transport", ["MEGABUS.COM", "MEGABUS UK"], (5, 40)),
    # eating out
    ("Nando's", "eating_out", ["NANDOS 0412", "NANDO'S CHICKENLAND"], (9, 45)),
    ("Pret", "eating_out", ["PRET A MANGER", "PRET A MANGER (EUROPE)"], (3, 12)),
    ("Deliveroo", "eating_out", ["DELIVEROO.CO.UK", "DELIVEROO"], (12, 45)),
    ("Greggs", "eating_out", ["GREGGS PLC", "GREGGS 1823"], (2, 9)),
    ("Costa", "eating_out", ["COSTA COFFEE 43012", "COSTA LTD"], (2, 9)),
    ("Starbucks", "eating_out", ["STARBUCKS 1203", "STARBUCKS COFFEE"], (2, 9)),
    ("Caffe Nero", "eating_out", ["CAFFE NERO", "NERO 0472"], (2, 9)),
    ("McDonald's", "eating_out", ["MCDONALDS 0982", "MCDONALD'S RESTAURANTS"], (3, 15)),
    ("KFC", "eating_out", ["KFC 3341", "KFC UK"], (4, 18)),
    ("Burger King", "eating_out", ["BURGER KING 4410", "BK 4410"], (4, 15)),
    ("Wagamama", "eating_out", ["WAGAMAMA LTD", "WAGAMAMA 0147"], (10, 50)),
    ("Pizza Express", "eating_out", ["PIZZAEXPRESS", "PIZZA EXPRESS 0112"], (12, 60)),
    ("Domino's", "eating_out", ["DOMINOS PIZZA", "DOMINO'S PIZZA 2281"], (10, 35)),
    ("Just Eat", "eating_out", ["JUST EAT.CO.UK", "JUSTEAT"], (12, 45)),
    ("Uber Eats", "eating_out", ["UBER *EATS", "UBER EATS"], (12, 45)),
    ("Leon", "eating_out", ["LEON RESTAURANTS", "LEON 0042"], (5, 15)),
    ("Itsu", "eating_out", ["ITSU 0091", "ITSU LTD"], (5, 15)),
    ("Wetherspoon", "eating_out", ["JD WETHERSPOON", "WETHERSPOONS 0332"], (4, 40)),
    ("Subway", "eating_out", ["SUBWAY 22931", "SUBWAY UK"], (4, 12)),
    ("Five Guys", "eating_out", ["FIVE GUYS 0233", "FIVE GUYS UK"], (10, 30)),
    ("Dishoom", "eating_out", ["DISHOOM", "DISHOOM KINGS CROSS"], (20, 90)),
    # bills
    ("British Gas", "bills", ["BRITISH GAS", "BG SERVICES"], (40, 180)),
    ("Thames Water", "bills", ["THAMES WATER", "THAMES WATER UTIL"], (20, 70)),
    ("Virgin Media", "bills", ["VIRGIN MEDIA", "VIRGIN MEDIA PYMTS"], (25, 90)),
    ("EE", "bills", ["EE LIMITED", "EE & T-MOBILE"], (10, 60)),
    ("Octopus Energy", "bills", ["OCTOPUS ENERGY", "OCTOPUSENERGY.COM"], (40, 180)),
    ("Vodafone", "bills", ["VODAFONE LTD", "VODAFONE UK"], (10, 50)),
    ("O2", "bills", ["O2 UK", "TELEFONICA O2"], (10, 50)),
    ("Three", "bills", ["THREE.CO.UK", "HUTCHISON 3G UK"], (10, 40)),
    ("BT", "bills", ["BT GROUP PLC", "BT.COM"], (25, 70)),
    ("Sky", "bills", ["SKY DIGITAL", "SKY UK LTD"], (20, 90)),
    ("EDF Energy", "bills", ["EDF ENERGY", "EDF ENERGY CUSTOMERS"], (40, 180)),
    ("E.ON", "bills", ["E.ON NEXT", "EON ENERGY"], (40, 180)),
    ("OVO Energy", "bills", ["OVO ENERGY", "OVO ENERGY LTD"], (40, 180)),
    ("Bristol Water", "bills", ["BRISTOL WATER", "BRISTOL WATER PLC"], (15, 50)),
    ("Bristol City Council", "bills", ["BRISTOL CITY COUNCIL", "BRISTOL CC CTAX"], (90, 200)),
    ("TV Licence", "bills", ["TV LICENCE MBP", "TVL TV LICENCE"], (13, 15)),
    ("giffgaff", "bills", ["GIFFGAFF.COM", "GIFFGAFF"], (6, 25)),
    ("Direct Line", "bills", ["DIRECT LINE INS", "DIRECT LINE GROUP"], (20, 90)),
    # subscriptions
    ("Netflix", "subscriptions", ["NETFLIX.COM", "NETFLIX INTERNATIONAL"], (5, 18)),
    ("Spotify", "subscriptions", ["SPOTIFY UK", "SPOTIFY P1A2B3"], (6, 18)),
    ("Amazon Prime", "subscriptions", ["AMZNPRIME*AB12C", "AMAZON PRIME*UK"], (5, 95)),
    ("Disney+", "subscriptions", ["DISNEY PLUS", "DISNEYPLUS"], (5, 11)),
    ("Audible", "subscriptions", ["AUDIBLE UK", "AUDIBLE.CO.UK"], (8, 10)),
    ("YouTube Premium", "subscriptions", ["YOUTUBE PREMIUM", "GOOGLE*YOUTUBE PREMIUM"], (6, 20)),
    ("Apple", "subscriptions", ["APPLE.COM/BILL", "APPLE.COM/UK"], (1, 15)),
    ("Microsoft 365", "subscriptions", ["MICROSOFT*365", "MSFT*MICROSOFT 365"], (6, 10)),
    ("Adobe", "subscriptions", ["ADOBE *CREATIVE CLD", "ADOBE SYSTEMS"], (10, 60)),
    ("Now TV", "subscriptions", ["NOW TV", "NOWTV.COM"], (10, 35)),
    ("PureGym", "subscriptions", ["PUREGYM LTD", "PURE GYM"], (18, 35)),
    ("The Gym Group", "subscriptions", ["THE GYM GROUP", "GYM GROUP LTD"], (18, 30)),
    ("Patreon", "subscriptions", ["PATREON* MEMBERSHIP", "PATREON"], (3, 15)),
    ("OpenAI", "subscriptions", ["OPENAI *CHATGPT SUBSCR", "OPENAI"], (18, 20)),
    ("Dropbox", "subscriptions", ["DROPBOX*", "DROPBOX.COM"], (8, 12)),
    ("Duolingo", "subscriptions", ["DUOLINGO", "DUOLINGO INC"], (7, 12)),
    ("Headspace", "subscriptions", ["HEADSPACE", "HEADSPACE INC"], (5, 12)),
    ("Paramount+", "subscriptions", ["PARAMOUNT+", "PARAMOUNTPLUS"], (5, 10)),
    # shopping
    ("Amazon", "shopping", ["AMZN MKTP UK*2H3", "AMAZON.CO.UK*AB1CD", "AMZ*AMAZON.CO.UK"], (4, 250)),
    ("ASOS", "shopping", ["ASOS.COM LTD", "ASOS.COM"], (12, 150)),
    ("Argos", "shopping", ["ARGOS LTD", "ARGOS RETAIL"], (8, 200)),
    ("Boots", "shopping", ["BOOTS 1137", "BOOTS UK LTD"], (3, 60)),
    ("John Lewis", "shopping", ["JOHN LEWIS PLC", "JOHN LEWIS 0042"], (10, 300)),
    ("Primark", "shopping", ["PRIMARK STORES", "PRIMARK 0812"], (4, 60)),
    ("Next", "shopping", ["NEXT RETAIL", "NEXT DIRECTORY"], (10, 120)),
    ("H&M", "shopping", ["H&M 0412", "H & M HENNES"], (5, 80)),
    ("Zara", "shopping", ["ZARA UK", "ZARA.COM"], (10, 120)),
    ("Uniqlo", "shopping", ["UNIQLO EUROPE", "UNIQLO UK"], (10, 100)),
    ("IKEA", "shopping", ["IKEA LTD", "IKEA BRISTOL"], (5, 300)),
    ("B&Q", "shopping", ["B&Q 1123", "B & Q"], (5, 150)),
    ("Currys", "shopping", ["CURRYS PC WORLD", "CURRYS 3221"], (10, 600)),
    ("Superdrug", "shopping", ["SUPERDRUG 0421", "SUPERDRUG STORES"], (3, 40)),
    ("TK Maxx", "shopping", ["TK MAXX", "TJX UK"], (5, 80)),
    ("eBay", "shopping", ["EBAY O*12-34567", "PAYPAL *EBAY"], (3, 200)),
    ("Etsy", "shopping", ["ETSY.COM", "ETSY INC"], (5, 60)),
    ("Waterstones", "shopping", ["WATERSTONES", "WATERSTONES 0212"], (5, 40)),
    ("WHSmith", "shopping", ["WHSMITH 0221", "WH SMITH"], (2, 30)),
    ("Decathlon", "shopping", ["DECATHLON UK", "DECATHLON 0041"], (5, 150)),
    ("JD Sports", "shopping", ["JD SPORTS 0012", "JD SPORTS FASHION"], (10, 150)),
    # entertainment
    ("Odeon", "entertainment", ["ODEON CINEMAS", "ODEON BRISTOL"], (8, 30)),
    ("Steam", "entertainment", ["STEAM PURCHASE", "STEAMGAMES.COM 4259"], (3, 60)),
    ("Ticketmaster", "entertainment", ["TICKETMASTER UK", "TM *TICKETMASTER"], (20, 150)),
    ("Vue", "entertainment", ["VUE CINEMAS", "VUE BSL LTD"], (7, 25)),
    ("Cineworld", "entertainment", ["CINEWORLD", "CINEWORLD 0091"], (7, 25)),
    ("Everyman", "entertainment", ["EVERYMAN CINEMA", "EVERYMAN MEDIA"], (12, 40)),
    ("PlayStation", "entertainment", ["PLAYSTATIONNETWORK", "SONY INTERACTIVE"], (5, 70)),
    ("Xbox", "entertainment", ["MICROSOFT*XBOX", "XBOX LIVE"], (5, 70)),
    ("Nintendo", "entertainment", ["NINTENDO ESHOP", "NINTENDO OF EUROPE"], (5, 60)),
    ("See Tickets", "entertainment", ["SEETICKETS.COM", "SEE TICKETS"], (15, 120)),
    ("Eventbrite", "entertainment", ["EVENTBRITE", "EB *EVENTBRITE"], (5, 60)),
    ("Skiddle", "entertainment", ["SKIDDLE", "SKIDDLE LTD"], (10, 60)),
    ("Hollywood Bowl", "entertainment", ["HOLLYWOOD BOWL", "HOLLYWOOD BOWL 0031"], (8, 40)),
    ("National Trust", "entertainment", ["NATIONAL TRUST", "NATIONAL TRUST ENT"], (8, 30)),
    ("Dice", "entertainment", ["DICE.FM", "DICE FM LTD"], (10, 60)),
    ("Merlin Entertainments", "entertainment", ["MERLIN ENTERTAINMENTS", "MERLIN ATTRACTIONS"], (20, 120)),
    # cash: one canonical merchant, so the bank wordings are what gets split
    ("ATM", "cash", ["CASH WITHDRAWAL LNK", "ATM WDL NATWEST", "CASH BARCLAYS ATM", "ATM LLOYDS BANK"], (10, 200)),
    ("ATM", "cash", ["LINK ATM SANTANDER", "CASH HALIFAX ATM"], (10, 200)),
    ("ATM", "cash", ["CASH HSBC ATM", "NOTEMACHINE ATM", "ATM POST OFFICE"], (10, 200)),
]

#----------EXTRA_MERCHANTS (v3 recipe)----------#

# Extra merchants for the v3 recipe. They only ever go to train/valid, so the unseen test is unchanged.
CAT_RANGE = {"groceries": (3, 90), "transport": (2, 120), "eating_out": (3, 40), "bills": (10, 150),
             "subscriptions": (3, 25), "shopping": (5, 150), "entertainment": (5, 80), "cash": (10, 200)}
EXTRA_MERCHANTS = [(name, cat, raws, CAT_RANGE[cat]) for name, cat, raws in [
    ("Londis", "groceries", ["LONDIS 0332", "LONDIS STORES"]),
    ("Nisa", "groceries", ["NISA LOCAL", "NISA 2219"]),
    ("Heron Foods", "groceries", ["HERON FOODS", "HERON FOODS 0101"]),
    ("One Stop", "groceries", ["ONE STOP 2231", "ONE STOP STORES"]),
    ("Amazon Fresh", "groceries", ["AMAZON FRESH", "AMZN FRESH UK"]),
    ("Abel & Cole", "groceries", ["ABEL & COLE", "ABEL AND COLE LTD"]),
    ("CrossCountry", "transport", ["CROSSCOUNTRY TRAINS", "XC TRAINS"]),
    ("Southern Railway", "transport", ["SOUTHERN RAILWAY", "SOUTHERNRAIL.COM"]),
    ("Northern", "transport", ["NORTHERN TRAINS", "NORTHERNRAILWAY"]),
    ("ScotRail", "transport", ["SCOTRAIL", "SCOTRAIL TICKETS"]),
    ("Addison Lee", "transport", ["ADDISON LEE", "ADDISONLEE.COM"]),
    ("FreeNow", "transport", ["FREENOW*", "FREE NOW UK"]),
    ("Santander Cycles", "transport", ["SANTANDER CYCLES", "TFL CYCLE HIRE"]),
    ("Texaco", "transport", ["TEXACO 1123", "TEXACO SERVICE STN"]),
    ("Heathrow Express", "transport", ["HEATHROW EXPRESS", "HEX TICKETS"]),
    ("British Airways", "transport", ["BRITISH AIRWAYS", "BA.COM"]),
    ("Zipcar", "transport", ["ZIPCAR UK", "ZIPCAR"]),
    ("Gail's", "eating_out", ["GAILS BAKERY", "GAIL'S 0412"]),
    ("Wasabi", "eating_out", ["WASABI 0112", "WASABI SUSHI"]),
    ("Tortilla", "eating_out", ["TORTILLA 0045", "TORTILLA MEX GRILL"]),
    ("Honest Burgers", "eating_out", ["HONEST BURGERS", "HONEST BURGER 07"]),
    ("Zizzi", "eating_out", ["ZIZZI 0412", "ZIZZI RISTORANTE"]),
    ("Pizza Hut", "eating_out", ["PIZZA HUT UK", "PIZZAHUT 1102"]),
    ("Taco Bell", "eating_out", ["TACO BELL 0021", "TACO BELL UK"]),
    ("Krispy Kreme", "eating_out", ["KRISPY KREME", "KRISPY KREME UK"]),
    ("Tim Hortons", "eating_out", ["TIM HORTONS", "TIM HORTONS 0091"]),
    ("Franco Manca", "eating_out", ["FRANCO MANCA", "FRANCO MANCA 22"]),
    ("Severn Trent", "bills", ["SEVERN TRENT WATER", "SEVERN TRENT"]),
    ("Wessex Water", "bills", ["WESSEX WATER", "WESSEX WATER SVCS"]),
    ("Utilita", "bills", ["UTILITA ENERGY", "UTILITA"]),
    ("Hyperoptic", "bills", ["HYPEROPTIC LTD", "HYPEROPTIC"]),
    ("Plusnet", "bills", ["PLUSNET PLC", "PLUSNET"]),
    ("TalkTalk", "bills", ["TALKTALK", "TALKTALK TELECOM"]),
    ("Admiral", "bills", ["ADMIRAL INSURANCE", "ADMIRAL"]),
    ("Lebara", "bills", ["LEBARA MOBILE", "LEBARA"]),
    ("Crunchyroll", "subscriptions", ["CRUNCHYROLL", "CRUNCHYROLL*SUB"]),
    ("Deezer", "subscriptions", ["DEEZER", "DEEZER UK"]),
    ("Strava", "subscriptions", ["STRAVA", "STRAVA INC"]),
    ("Calm", "subscriptions", ["CALM.COM", "CALM APP"]),
    ("Canva", "subscriptions", ["CANVA*", "CANVA PTY"]),
    ("Notion", "subscriptions", ["NOTION LABS", "NOTION.SO"]),
    ("GitHub", "subscriptions", ["GITHUB INC", "GITHUB.COM"]),
    ("Kindle Unlimited", "subscriptions", ["KINDLE UNLTD", "AMZN DIGITAL*KINDLE"]),
    ("BritBox", "subscriptions", ["BRITBOX", "BRITBOX INTL"]),
    ("Sports Direct", "shopping", ["SPORTS DIRECT", "SPORTSDIRECT.COM"]),
    ("Matalan", "shopping", ["MATALAN", "MATALAN RETAIL"]),
    ("Lush", "shopping", ["LUSH RETAIL", "LUSH 0221"]),
    ("Selfridges", "shopping", ["SELFRIDGES & CO", "SELFRIDGES"]),
    ("Dunelm", "shopping", ["DUNELM 0221", "DUNELM LTD"]),
    ("Screwfix", "shopping", ["SCREWFIX DIRECT", "SCREWFIX"]),
    ("The Works", "shopping", ["THE WORKS STORES", "THE WORKS"]),
    ("Hobbycraft", "shopping", ["HOBBYCRAFT", "HOBBYCRAFT 0042"]),
    ("Smyths Toys", "shopping", ["SMYTHS TOYS", "SMYTHS"]),
    ("AliExpress", "shopping", ["ALIEXPRESS", "ALIEXPRESS.COM"]),
    ("Vinted", "shopping", ["VINTED", "VINTED UAB"]),
    ("Picturehouse", "entertainment", ["PICTUREHOUSE", "PICTUREHOUSE CINEMAS"]),
    ("Showcase Cinemas", "entertainment", ["SHOWCASE CINEMAS", "SHOWCASE 0021"]),
    ("Epic Games", "entertainment", ["EPIC GAMES STORE", "EPICGAMES.COM"]),
    ("Fever", "entertainment", ["FEVER UP", "FEVERUP.COM"]),
    ("Bowlplex", "entertainment", ["BOWLPLEX", "BOWLPLEX 0011"]),
    ("Puttshack", "entertainment", ["PUTTSHACK", "PUTTSHACK BRISTOL"]),
    ("Flip Out", "entertainment", ["FLIP OUT", "FLIP OUT ARENA"]),
    ("Tate", "entertainment", ["TATE ENTERPRISES", "TATE MODERN"]),
    ("English Heritage", "entertainment", ["ENGLISH HERITAGE", "ENGLISH HERITAGE ENT"]),
    ("ATM", "cash", ["ATM TSB BANK", "CASH MONZO ATM", "ATM CO-OP BANK"]),
]]

#----------AUGMENTATION----------#

# Augmentation for train/valid lines. The unseen test's own codes are excluded so nothing leaks.
TEST_ONLY_CODES = {"BCC", "VIS", "POS"}
NAME_SUFFIXES = [" LTD", " PLC", " UK", " GB", ".COM", " {n}", " {n}", " STORE {n}"]


#-----random_code-----#
def random_code(rng: random.Random) -> str:
    while True:
        code = "".join(rng.choice("ABCDEFGHIJKLMNOPRSTUVWXY") for _ in range(rng.randint(2, 4)))
        if code not in TEST_ONLY_CODES:
            return code


#-----augment_raw-----#
def augment_raw(rng: random.Random, raw: str) -> str:
    """Store numbers and corporate suffixes the model should learn to drop."""
    if rng.random() < 0.3:
        raw += rng.choice(NAME_SUFFIXES).format(n=rng.randint(10, 9999))
    return raw


#----------LINE_FORMATS----------#

CITIES = ["BRISTOL", "LONDON", "BATH", "CARDIFF", "MANCHESTER", "LEEDS", "GLASGOW", ""]

TRAIN_TEMPLATES = [
    "{raw} {city} {date} £{amount}",
    "CARD PAYMENT TO {raw} ON {date} {amount}",
    "{raw} {ref} {amount} GBP",
    "CONTACTLESS {raw} {city} £{amount}",
    "DD {raw} REF {ref} £{amount}",
    "{date} {raw} {city} -{amount}",
    "DEB {raw} {city} £{amount}",
    "FPO {raw} {ref} {amount}",
    "{raw} CD {card} {date} {amount}",
    "APPLE PAY {raw} {city} £{amount}",
    "SO {raw} {date} {amount}",
]
UNSEEN_TEMPLATES = [  # only ever used for the unseen test
    "POS {date} {raw} {city} GBP {amount}",
    "BCC {raw} {ref} {city} £{amount}",
    "{date} DEB {raw} £{amount}",
    "VIS {raw} {city} {date} {amount}",
]


#----------GENERATION----------#

#-----split_merchants-----#
def split_merchants(rng: random.Random, valid_frac: float, test_frac: float) -> dict[str, list]:
    """Stratified by category: every split gets merchants from every category."""
    by_cat = defaultdict(list)
    for entry in MERCHANTS:
        by_cat[entry[1]].append(entry)
    splits = {"train": [], "valid": [], "test": []}
    for entries in by_cat.values():
        entries = entries[:]
        rng.shuffle(entries)
        n_test = max(1, round(test_frac * len(entries)))
        n_valid = max(1, round(valid_frac * len(entries)))
        splits["test"] += entries[:n_test]
        splits["valid"] += entries[n_test:n_test + n_valid]
        splits["train"] += entries[n_test + n_valid:]
    return splits


#-----noisy_raw-----#
def noisy_raw(rng: random.Random, raw: str) -> str:
    """Real statements truncate merchant text and sometimes aren't upper case."""
    if len(raw) > 18 and rng.random() < 0.3:
        raw = raw[:18].rstrip()
    if rng.random() < 0.1:
        raw = raw.title()
    return raw


#-----make_example-----#
def make_example(rng: random.Random, merchants: list, templates: list, augment: bool = False) -> dict:
    # pick a category first, then a merchant in it, so categories stay balanced (cash has only one merchant)
    category = rng.choice(sorted({m[1] for m in merchants}))
    merchant, _, raws, (low, high) = rng.choice([m for m in merchants if m[1] == category])
    amount = round(rng.uniform(low, high), 2)
    amount_text = f"{amount:,.2f}" if rng.random() < 0.5 else f"{amount:.2f}"
    raw = rng.choice(raws)
    if augment:
        raw = augment_raw(rng, raw)
    line = rng.choice(templates).format(
        raw=noisy_raw(rng, raw),
        city=rng.choice(CITIES),
        date=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}",
        ref=f"{rng.randint(10000, 99999)}",
        card=f"{rng.randint(1000, 9999)}",
        amount=amount_text,
    )
    if augment and rng.random() < 0.3:
        line = f"{random_code(rng)} {line}"  # unknown bank code in front: it is not the merchant
    line = " ".join(line.split())  # collapse double spaces from empty cities
    return {"line": line, "label": {"merchant": merchant, "amount": amount, "category": category}}


#-----write_jsonl-----#
def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


#----------MAIN----------#
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--valid", type=int, default=120)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--test-unseen", type=int, default=150)
    parser.add_argument("--valid-frac", type=float, default=0.15, help="share of merchants held out for validation")
    parser.add_argument("--test-frac", type=float, default=0.2, help="share of merchants held out for the unseen test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    parser.add_argument("--no-extra", action="store_true", help="v2 recipe: no extra train/valid merchants")
    parser.add_argument("--no-augment", action="store_true", help="v2 recipe: no prefix/suffix augmentation")
    args = parser.parse_args()

    splits = split_merchants(random.Random(args.seed), args.valid_frac, args.test_frac)
    seen_merchants = splits["train"][:]          # the "seen" test always uses the original train merchants
    if not args.no_extra:
        extra = EXTRA_MERCHANTS[:]
        random.Random(f"{args.seed}-extra").shuffle(extra)
        n_valid = round(0.2 * len(extra))
        splits["valid"] += extra[:n_valid]
        splits["train"] += extra[n_valid:]

    # one random stream per file: test files don't change when the training recipe does
    stream = lambda name: random.Random(f"{args.seed}-{name}")  # noqa: E731
    aug = not args.no_augment
    r_train, r_valid, r_test, r_unseen = stream("train"), stream("valid"), stream("test"), stream("test_unseen")
    files = {
        "train": [make_example(r_train, splits["train"], TRAIN_TEMPLATES, aug) for _ in range(args.train)],
        "valid": [make_example(r_valid, splits["valid"], TRAIN_TEMPLATES, aug) for _ in range(args.valid)],
        "test": [make_example(r_test, seen_merchants, TRAIN_TEMPLATES) for _ in range(args.test)],
        "test_unseen": [make_example(r_unseen, splits["test"], UNSEEN_TEMPLATES) for _ in range(args.test_unseen)],
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, rows in files.items():
        write_jsonl(rows, out / f"{name}.jsonl")
    (out / "splits.json").write_text(json.dumps(
        {name: sorted({e[0] for e in entries}) for name, entries in splits.items()}, indent=2))

    print(f"Merchants: train {len(splits['train'])}, valid {len(splits['valid'])}, test {len(splits['test'])}")
    print("Rows:", {name: len(rows) for name, rows in files.items()})
    print("Train categories:", dict(sorted(Counter(r["label"]["category"] for r in files["train"]).items())))
    print("Unseen test merchants:", ", ".join(sorted({e[0] for e in splits["test"]})))
    for row in files["train"][:2] + files["test_unseen"][:2]:
        print(" ", row["line"], "->", row["label"])


if __name__ == "__main__":
    main()
