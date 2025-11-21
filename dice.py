import re
import random

# ---------- Dice roller ----------
DICE_RE = re.compile(r"(?P<count>\d*)d(?P<sides>\d+)(?P<mods>(?:[+-]\d+)*)")
MOD_RE = re.compile(r"[+-]\d+")

def roll_dice(expr: str) -> dict:
    expr = expr.replace(" ", "")
    m = DICE_RE.match(expr)
    if not m:
        if expr.isdigit() or (expr.startswith("-") and expr[1:].isdigit()):
            return {"rolls": [], "total": int(expr)}
        parts = re.findall(r"[+-]?[^+-]+", expr)
        total = 0
        breakdown = []
        for p in parts:
            part = roll_dice(p)
            breakdown.append(part)
            total += part["total"]
        return {"breakdown": breakdown, "total": total}

    count = int(m.group("count") or 1)
    sides = int(m.group("sides"))
    mods = m.group("mods") or ""

    rolls = [random.randint(1, sides) for _ in range(count)]
    mod_total = sum(int(x) for x in MOD_RE.findall(mods)) if mods else 0
    return {"rolls": rolls, "total": sum(rolls) + mod_total}