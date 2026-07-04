"""Hypothesis registry helpers: list registered hypotheses and their status."""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HYP = os.path.join(ROOT, "hypotheses")


def list_hypotheses() -> list[dict]:
    out = []
    for f in sorted(os.listdir(HYP)):
        if not f.endswith(".md"):
            continue
        text = open(os.path.join(HYP, f)).read()
        status = re.search(r"\*\*Status\*\*:\s*(\S+)", text)
        title = re.search(r"^# (.+)$", text, re.M)
        out.append({"file": f, "title": title.group(1) if title else f,
                    "status": status.group(1) if status else "?"})
    return out


if __name__ == "__main__":
    for h in list_hypotheses():
        print(f"{h['status']}  {h['title']}   ({h['file']})")
