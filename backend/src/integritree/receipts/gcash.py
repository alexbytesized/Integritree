"""Conservative OCR field candidates for the local benchmark, not confirmed inputs."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re

PARSER_VERSION = "gcash_candidates_v1"
AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}(?!\d)")
DATE = re.compile(
    r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})\s+(\d{1,2}):([0-5]\d)\s*([AP])\.?M\.?\b",
    re.I,
)


def parse_candidates(lines):
    texts = [line["text"].strip() for line in lines]
    text = "\n".join(texts)
    lower = text.lower()
    workflows = []
    if "express send" in lower:
        workflows.append(("express_send", "TRANSFER"))
    if "bank transfer" in lower:
        workflows.append(("bank_transfer", "DEBIT"))
    if "pay online" in lower or "paid and linked via gcash" in lower:
        workflows.append(("pay_online", "PAYMENT"))
    if "pay qr" in lower or "scan to pay" in lower:
        workflows.append(("merchant_qr", "PAYMENT"))
    dates, amounts, references = set(), set(), set()
    for index, line in enumerate(texts):
        # Principal labels only. Never substitute fee, balance, or total.
        if re.match(r"^(?:transfer\s+)?amount\b", line, re.I):
            found = AMOUNT.findall(line)
            if (
                not found
                and index + 1 < len(texts)
                and re.fullmatch(r"[₱P\s]*[\d,.]+", texts[index + 1])
            ):
                found = AMOUNT.findall(texts[index + 1])
            for value in found:
                try:
                    amounts.add(format(Decimal(value.replace(",", "")), ".2f"))
                except InvalidOperation:
                    pass
        match = re.search(
            r"\bref(?:erence)?\s*(?:no\.?|number)\s*[:.]?\s*(\d[\d ]*)", line, re.I
        )
        if match:
            references.add(match[1].replace(" ", ""))
        elif re.fullmatch(
            r"ref(?:erence)?\s*(?:no\.?|number)\s*[:.]?", line, re.I
        ) and index + 1 < len(texts):
            if re.fullmatch(r"\d+", texts[index + 1]):
                references.add(texts[index + 1])
    for match in DATE.finditer(text):
        month, day, year, hour, minute, ap = match.groups()
        try:
            value = datetime.strptime(
                f"{month[:3]} {day} {year} {hour}:{minute} {ap.upper()}M",
                "%b %d %Y %I:%M %p",
            )
            dates.add(value.isoformat(timespec="minutes"))
        except ValueError:
            pass
    unique = lambda values: next(iter(values)) if len(values) == 1 else None
    workflow = workflows[0] if len(workflows) == 1 else (None, None)
    stamp = unique(dates)
    fields = {
        "workflow": workflow[0],
        "category": workflow[1],
        "amount": unique(amounts),
        "date": stamp[:10] if stamp else None,
        "time": stamp[11:] if stamp else None,
        "reference": unique(references),
    }
    return {
        "parser_version": PARSER_VERSION,
        "fields": fields,
        "missing_fields": [key for key, value in fields.items() if value is None],
        "ambiguous_fields": [
            name
            for name, values in (
                ("amount", amounts),
                ("datetime", dates),
                ("reference", references),
                ("workflow", workflows),
            )
            if len(values) > 1
        ],
        "requires_confirmation": True,
        "note": "Candidate text does not verify payment status, funding, account roles, or authenticity.",
    }
