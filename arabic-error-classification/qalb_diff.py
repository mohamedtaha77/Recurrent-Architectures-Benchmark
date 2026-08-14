# -*- coding: utf-8 -*-
"""
Tokenize a QALB sentence pair, diff it into edits, and type each edit
(hamza_alef, alef_maqsura, ta_marbuta, punctuation, ...).

Trimmed and consolidated from a separate Arabic grammar-correction project
of mine (arabic-gec-starter: src/errors.py + src/evaluate.py), where this
logic was built and checked against real QALB character confusions. Only
what prep_data.py here actually needs is kept: no synthetic error
injection, no evaluation scoring, just tokenize + extract_edits + classify.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Character inventory
# ---------------------------------------------------------------------------
ALEF_HAMZA = "أإآٱ"
ALEF_BARE = "ا"
TA_MARBUTA = "ة"
HA = "ه"
ALEF_MAQSURA = "ى"
YA = "ي"
WAW = "و"
HAMZA_SEATS = "أإؤئءآ"

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SUKUN, SHADDA = "ْ", "ّ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
SHORT_VOWELS = FATHA + DAMMA + KASRA
TANWEEN = FATHATAN + DAMMATAN + KASRATAN

# Same rasm (skeleton), different dot placement.
RASM = {
    "ب": "تثني", "ت": "بثني", "ث": "بتني", "ن": "بتثي", "ي": "بتثن",
    "ج": "حخ", "ح": "جخ", "خ": "جح",
    "د": "ذ", "ذ": "د", "ر": "ز", "ز": "ر",
    "س": "ش", "ش": "س", "ص": "ض", "ض": "ص",
    "ط": "ظ", "ظ": "ط", "ع": "غ", "غ": "ع", "ف": "ق", "ق": "ف",
}

# Letters that sound identical in spoken Arabic dialects.
PHONETIC = {
    "ص": "س", "س": "صز", "ط": "ت", "ت": "طث", "ض": "ظد", "ظ": "ضزذ",
    "د": "ضذ", "ذ": "زدظ", "ز": "ذظ", "ث": "سـت",
    "ق": "ءأك", "ء": "قأ",
    "ح": "ه", "ه": "ح", "ع": "اء", "خ": "غ", "غ": "خع",
    "ك": "ق", "ج": "ز",
}

PUNCT = set("،؛؟.!:,;?\"'()[]-–—")

_TOK = re.compile(r"[ء-يً-ْٰـ]+|[0-9]+|[^\sء-ي0-9]")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text)


# ---------------------------------------------------------------------------
# Alignment -> edits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Edit:
    i: int
    j: int
    repl: tuple[str, ...]

    @property
    def kind(self) -> str:
        return ("INS" if self.i == self.j else
                "DEL" if not self.repl else "SUB")


def _align(a: list[str], b: list[str]) -> list[tuple[str, int, int]]:
    """Levenshtein backtrace. Returns ops as (tag, i, j) over a and b."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("M" if a[i - 1] == b[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", i - 1, j))
            i -= 1
        else:
            ops.append(("I", i, j - 1))
            j -= 1
    return ops[::-1]


def extract_edits(src: list[str], tgt: list[str]) -> set[Edit]:
    """
    Group non-match ops into edits. Adjacent substitutions stay separate
    (two misspelled words are two errors, not one); only buffers containing
    an insert/delete get merged, since those are the genuine merge/split
    cases.
    """
    ops = _align(src, tgt)
    edits, buf = set(), []

    def flush():
        if not buf:
            return
        if all(o[0] == "S" for o in buf):
            for o in buf:
                edits.add(Edit(o[1], o[1] + 1, (tgt[o[2]],)))
            buf.clear()
            return
        si = [o for o in buf if o[0] in "SD"]
        ti = [o for o in buf if o[0] in "SI"]
        i = si[0][1] if si else buf[0][1]
        j = si[-1][1] + 1 if si else i
        repl = tuple(tgt[o[2]] for o in ti)
        edits.add(Edit(i, j, repl))
        buf.clear()

    for op in ops:
        if op[0] == "M":
            flush()
        else:
            buf.append(op)
    flush()
    return edits


# ---------------------------------------------------------------------------
# Edit typing
# ---------------------------------------------------------------------------
def _endswap(a: str, b: str, x: str, y: str) -> bool:
    return a[:-1] == b[:-1] and {a[-1:], b[-1:]} == {x, y}


def classify(src: list[str], e: Edit) -> str:
    o = " ".join(src[e.i:e.j])
    n = " ".join(e.repl)
    if (o and all(c in PUNCT for c in o.replace(" ", ""))) or \
       (n and all(c in PUNCT for c in n.replace(" ", ""))):
        return "punctuation"
    if e.kind == "SUB" and e.j - e.i == 1 and len(e.repl) == 1:
        if _endswap(o, n, TA_MARBUTA, HA):
            return "ta_marbuta"
        if _endswap(o, n, ALEF_MAQSURA, YA):
            return "alef_maqsura"
        if o and n and len(o) == len(n):
            diff = [(x, y) for x, y in zip(o, n) if x != y]
            if len(diff) == 1:
                x, y = diff[0]
                if x in ALEF_HAMZA + ALEF_BARE and y in ALEF_HAMZA + ALEF_BARE:
                    return "hamza_alef"
                if x in HAMZA_SEATS + WAW + YA and y in HAMZA_SEATS + WAW + YA:
                    return "hamza_seat"
                if x in SHORT_VOWELS + SUKUN + SHADDA or y in SHORT_VOWELS + SUKUN + SHADDA:
                    return "harakat"
                if x in TANWEEN or y in TANWEEN:
                    return "tanween"
                if y in RASM.get(x, ""):
                    return "rasm_dots"
                if y in PHONETIC.get(x, ""):
                    return "phonetic"
                return "typo"
            return "multi_char"
        if o and n and abs(len(o) - len(n)) == 1:
            longer, shorter = (o, n) if len(o) > len(n) else (n, o)
            ch = next((c for c in longer if longer.replace(c, "", 1) == shorter), None)
            if ch is None:
                return "spelling/morph"
            if ch in SHORT_VOWELS + SUKUN + SHADDA:
                return "harakat"
            if ch in TANWEEN:
                return "tanween"
            if ch in HAMZA_SEATS:
                return "hamza_seat"
            if ch == ALEF_BARE and (o.endswith(WAW + ALEF_BARE) or n.endswith(WAW + ALEF_BARE)):
                return "alef_fariqa"
            return "char_ins_del"
        return "spelling/morph"
    if e.j - e.i > 1 and len(e.repl) == 1:
        return "merge"
    if e.j - e.i == 1 and len(e.repl) > 1:
        return "split"
    return "other"
