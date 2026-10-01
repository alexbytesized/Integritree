"""Conservative supported-layout gate, separate from editable OCR field candidates.

Text anchors recognize sampled GCash layouts, not image authenticity. A missing or
conflicting heading/layout stays unsupported even if a client supplies valid fields.
"""

import re


def identify_layout(lines):
    texts = [row["text"].strip().lower() for row in lines]
    text = "\n".join(texts)
    if re.search(
        r"\b(failed|unsuccessful|cancelled|canceled|pending|processing)\b", text
    ):
        return None, None, "Only completed transaction receipts are supported."
    amount = any(re.match(r"^(transfer\s+)?amount\b", line) for line in texts)
    reference = bool(re.search(r"\bref(?:erence)?\s*(?:no\b|number\b)", text))
    date_anchor = (
        "transfer date" in text
        or any(line.startswith("date") for line in texts)
        or bool(re.search(r"\b[a-z]{3,9}\s+\d{1,2},?\s+20\d{2}", text))
    )
    layouts = []
    if (
        "express send" in text
        and "sent via gcash" in text
        and "total amount sent" in text
    ):
        layouts.append(("gcash_express_send_v1", "personal_wallet"))
    if ("pay online" in text or "paid and linked via gcash" in text) and re.search(
        r"paid (?:and linked )?via gcash", text
    ):
        layouts.append(("gcash_pay_online_v1", "merchant"))
    if (
        "bank transfer complete" in text
        and "transfer date" in text
        and "transfer amount" in text
    ):
        # Recipient bank is a separate label/block; do not search ads or names for roles.
        bank = []
        for i, line in enumerate(texts):
            if re.match(r"^bank(?:\s|$)", line) and not line.startswith(
                "bank transfer"
            ):
                bank.append(line)
                for following in texts[i + 1 : i + 4]:
                    if re.match(r"^(account|receipt|transfer)\b", following):
                        break
                    bank.append(following)
        if not bank:
            return (
                None,
                None,
                "The destination bank could not be identified. Upload a clearer receipt.",
            )
        destination = " ".join(bank)
        if (
            "wallet" in destination
            or "maya philippines" in destination
            or "vybe" in destination
        ):
            return (
                None,
                "personal_wallet",
                "This bank-transfer screen names a wallet or mixed destination. It is outside the current bank-account workflow.",
            )
        layouts.append(("gcash_bank_transfer_v1", None))
    if len(layouts) != 1 or sum((amount, reference, date_anchor)) < 2:
        return (
            None,
            None,
            "The receipt layout is unclear or unsupported. Use a complete Express Send, Pay Online, or bank-account transfer screenshot.",
        )
    return *layouts[0], None
