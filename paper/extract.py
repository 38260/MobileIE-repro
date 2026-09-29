"""Extract the ICCV MobileIE PDF into an ordered item list for a bilingual rebuild.

    python paper/extract.py            # writes paper/bilingual/items.json + assets/*.png
    python paper/extract.py --inspect  # classification only, no PNGs

Reading order: per page, a full-width block is emitted on its own and resets the
two-column run; otherwise the left column is emitted before the right one.
Figures and tables are cropped from the live page (3x) instead of being rebuilt, so
vector plots, sub-caption grids and table rules come through exactly as printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
PDF = next(ROOT.parent.glob("Yan_MobileIE_*.pdf"))
OUT = ROOT / "bilingual"
ASSETS = OUT / "assets"

MID = 306.0            # column boundary of the 612pt page
FLOAT_GAP = 19.0       # vertical pull range for float growth
CAPTION_RE = re.compile(r"(?m)^\s*((?:Figure|Table)\s+\d+)\.\s")
HEAD_RE = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-Z]")
MATH_RE = re.compile(r"\\label|\\frac|\\left|\\right|\\sum|\\mu|\\sigma|\\mathcal|_\{|\^\{"
                     r"|\\geq|\\leq|\\odot|\\times|\\cdot|&\\|\{$|^\}$|\\rm|\\Arrowvert|\\Delta")


_TOKENS: set[str] = set()
DECISIONS: list[tuple[str, str, bool]] = []
# the vocabulary test misses a word that appears exactly once, broken across a line,
# and compounds whose left half never appears unbroken anywhere
HYPHEN_KEEP = {("the", "art"), ("dslr", "comparable"), ("laplace", "like"),
               ("semi", "supervised"), ("star", "shaped")}
HYPHEN_GLUE = {("mini", "malism")}
SUFFIXES = {"ing", "ed", "al", "ly", "ion", "es", "s", "ness", "ment", "ance",
            "ence", "tion", "ful", "ive", "ous", "able", "ity"}


def build_tokens(doc):
    """Words of the paper, excluding the halves of line-end hyphen breaks.

    Without this, 'ad-\\njusting' would make both 'ad' and 'justing' look like words and
    every broken word would keep its hyphen.
    """
    global _TOKENS
    raw = "\n".join(page.get_text() for page in doc)
    total = Counter(re.findall(r"[a-z]+", re.sub(r"-\n", " ", raw).lower()))
    from_break: Counter = Counter()
    for m in re.finditer(r"([A-Za-z]+)-\n([a-zA-Z]+)", raw):
        from_break[m.group(1).lower()] += 1
        from_break[m.group(2).lower()] += 1
    _TOKENS = {t for t, n in total.items() if n > from_break.get(t, 0)}


def norm(text: str) -> str:
    """Join PDF line breaks into flowing text, undoing end-of-line hyphenation.

    Glue when the joined form is a word the paper actually uses ('net-' + 'work'),
    keep the hyphen when it is not ('resource-' + 'constrained').
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-"):
            prev = re.split(r"[\s(,.;:/\[\]-]+", out[:-1])[-1].lower()
            nxt = re.split(r"[\s(,.;:/\]\-]+", ln)[0].lower()
            glued = prev + nxt
            keep = (prev in _TOKENS and nxt in _TOKENS
                    and nxt not in SUFFIXES and glued not in _TOKENS)
            if (prev, nxt) in HYPHEN_KEEP:
                keep = True
            if (prev, nxt) in HYPHEN_GLUE:
                keep = False
            DECISIONS.append((prev, nxt, keep))
            out = out[:-1] + ("-" if keep else "") + ln
        else:
            out = out + " " + ln
    return re.sub(r"\s+", " ", out).strip()


def col_of(x0: float, x1: float) -> int:
    if x0 >= MID - 8:
        return 1
    if x1 <= MID + 8:
        return 0
    return -1


def is_prose(rect, text: str) -> bool:
    """Body paragraph, as opposed to table cells or figure labels.

    Length and height alone are not enough: a two-sentence paragraph is shorter than a
    table body, and single-column tables are as wide as text. Sentence boundaries and
    digit density separate them reliably.
    """
    if rect.width > 300 or len(text) < 120:
        return False
    if re.search(r"[a-z]\.\s+[A-Z]", text):
        return True
    digits = sum(c.isdigit() for c in text) / len(text)
    return text.endswith(".") and len(text.split()) > 18 and digits < 0.08


def is_math(text: str) -> bool:
    body = norm(text)
    if not body:
        return False
    if re.fullmatch(r"\(\s*\d+\s*\)", body):
        return True
    squashed = re.sub(r"\s+", "", body)          # PDF splits '\label' into '\ l a b e l'
    if MATH_RE.search(squashed):
        return True
    if squashed.isdigit():
        return False                             # running page number, not an equation
    if squashed and re.fullmatch(r"[\\{}()\[\}\d^_&|,.=]+", squashed):
        return True                              # brace / number debris of an equation
    letters = sum(ch.isalpha() for ch in body)
    return bool(letters) and ("=" in squashed) and len(body.split()) <= 6 and letters / len(body) < 0.5


def graphics_rects(page):
    out = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.width > 3 and r.height > 3 and not r.is_empty:
            out.append(r)
    for info in page.get_image_info():
        out.append(fitz.Rect(info["bbox"]))
    return out


def float_region(cap: fitz.Rect, cand: list[fitz.Rect], gfx: list[fitz.Rect]) -> fitz.Rect:
    """Grow a crop from the caption upwards over graphics and non-prose text."""
    seed = fitz.Rect(cap)
    limit = cap.y1
    for _ in range(90):
        grown = fitz.Rect(seed)
        probe_gfx = fitz.Rect(seed.x0 - 20, seed.y0 - 45, seed.x1 + 20, seed.y1 + 20)
        for r in gfx:
            if r.y1 <= limit + 2 and r.intersects(probe_gfx):
                grown |= r
        probe_txt = fitz.Rect(seed.x0 - 20, seed.y0 - FLOAT_GAP, seed.x1 + 20, seed.y1 + 8)
        for r in cand:
            if r.y1 > limit + 2:
                continue
            center = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            if probe_txt.contains(center) or (r.intersects(probe_txt) and seed.y0 - r.y1 < FLOAT_GAP):
                grown |= r
        if grown == seed or grown.height > 0.82 * 792:
            break
        seed = grown
    seed.y1 = max(seed.y1, limit)
    return seed


def _near(a, b, vy, vx) -> bool:
    return a.y0 - b.y1 < vy and b.y0 - a.y1 < vy and a.x0 - b.x1 < vx and b.x0 - a.x1 < vx


def math_regions(pairs):
    """Cluster equation fragments by proximity, then absorb the short orphan pieces
    that multi-line LaTeX leaves behind ('b e', 'gin', '='). Merging repeats because a
    group only reaches its neighbours once it has grown."""
    groups: list[fitz.Rect] = []
    for r in sorted([fitz.Rect(r) for r, t in pairs if is_math(t)], key=lambda r: r.y0):
        hits = [g for g in groups if _near(g, r, 14, 22)]
        for g in hits:
            groups.remove(g)
            r |= g
        groups.append(r)
    for _ in range(5):
        joined = False
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                if _near(a, b, 14, 22):
                    groups.remove(b)
                    groups[i] = fitz.Rect(a) | b
                    joined = True
                    break
            if joined:
                break
        if not joined:
            break

    loose = [fitz.Rect(r) for r, t in pairs
             if not is_math(t) and not is_prose(r, norm(t)) and len(norm(t)) <= 60]
    for r in loose:
        for g in groups:
            if r.y0 < g.y1 + 10 and r.y1 > g.y0 - 10 and r.x0 - g.x1 < 90 and g.x0 - r.x1 < 90:
                g |= r
                break
    return [g + (-3, -3, 3, 3) for g in groups]



def covered(rect, rects) -> bool:
    area = rect.get_area()
    if area <= 0:
        return False
    for other in rects:
        if other.contains(fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)):
            return True
        inter = fitz.Rect(other) & rect
        if not inter.is_empty and inter.get_area() / area > 0.55:
            return True
    return False


def classify(rect, text: str):
    body = norm(text)
    if not body:
        return "skip"
    if rect.y1 < 68 or re.fullmatch(r"\d{1,2}|\d{4,5}", body):
        return "skip"                       # running head / page number
    if body.startswith("arXiv:"):
        return "skip"
    if CAPTION_RE.search(body[:40]):
        return "caption"
    if HEAD_RE.match(body) and len(body) < 70:
        return "heading"
    if body in ("Abstract", "Acknowledgement", "References", "Supplementary Material"):
        return "heading"
    if body.startswith("*") or (rect.y0 > 640 and len(body) < 45):
        return "footnote"
    return "para"


def order(events):
    """events: (y, col, kind, payload) -> reading order with column runs."""
    pend = {0: [], 1: []}
    out = []

    def flush():
        for col in (0, 1):
            out.extend(pend[col])
            pend[col] = []

    for ev in sorted(events, key=lambda e: (round(e[0], 1), e[1])):
        if ev[1] == -1:
            flush()
            out.append(ev)
        else:
            pend[ev[1]].append(ev)
    flush()
    return out


def box(rect):
    return [round(v, 1) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]


def stitch(items):
    """Rejoin a paragraph that floats interrupt mid-sentence.

    The paper wraps '...diffusion pro-' around Figure 1 and continues with 'cesses ...'.
    A linear bilingual document reads the whole paragraph first, then shows the float.
    Merging repeats, because one paragraph can be cut more than once before it ends.
    """
    out, pos, n = [], 0, len(items)
    while pos < n:
        it = items[pos]
        if it["kind"] != "para":
            out.append(it)
            pos += 1
            continue
        merged, carried = dict(it), []
        pos += 1
        while not merged["text"].endswith((".", "!", "?", ":", ")")) and pos < n:
            pending = []
            while pos < n and items[pos]["kind"] != "para" and len(pending) < 4:
                pending.append(items[pos])
                pos += 1
            tail = merged["text"]
            head = items[pos]["text"].split(" ")[0] if pos < n else ""
            if pos >= n or items[pos]["kind"] != "para":
                carried.extend(pending)
                break
            if not (tail.endswith("-") or (head[:1].islower() and len(head) > 1)):
                carried.extend(pending)
                break
            merged["text"] = (tail[:-1] if tail.endswith("-") else tail + " ") + items[pos]["text"]
            carried.extend(pending)
            pos += 1
        out.append(merged)
        out.extend(carried)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    doc = fitz.open(PDF)
    build_tokens(doc)
    items = []
    if not args.inspect:
        ASSETS.mkdir(parents=True, exist_ok=True)

    for pno in range(doc.page_count):
        page = doc[pno]
        raw = [(fitz.Rect(b[:4]), b[4]) for b in page.get_text("blocks") if b[6] == 0]
        gfx = graphics_rects(page)

        # a caption can share its block with a sub-figure label, so anchor on the line
        captions = []
        for rect, text in raw:
            for match in CAPTION_RE.finditer(text):
                head = text[:match.start()].strip()
                if len(head) > 30:
                    continue      # '...as shown in\nTable 6. With minimal...' is prose flow
                cap = fitz.Rect(rect)
                if head:
                    hits = page.search_for(match.group(1).strip() + ".")
                    if hits:
                        cap.y0 = min(h.y0 for h in hits) - 1
                captions.append({"rect": cap, "text": norm(text[match.start():])})

        cand = [r for r, t in raw if not is_prose(r, norm(t)) and not CAPTION_RE.search(norm(t)[:40])]
        floats = [(cap, float_region(cap["rect"], cand, gfx)) for cap in captions]
        seeds = [rect for _, rect in floats]

        free = [(r, t) for r, t in raw if not covered(r, seeds)]
        eqs = math_regions(free)

        events = []
        for cap, rect in floats:
            kind = "figure" if cap["text"].lower().startswith("figure") else "table"
            events.append((cap["rect"].y0, col_of(cap["rect"].x0, cap["rect"].x1), kind,
                           {"caption": cap["text"], "rect": rect}))
        for rect in eqs:
            events.append((rect.y0, col_of(rect.x0, rect.x1), "equation",
                           {"rect": rect, "raw": norm(" ".join(
                               t for r, t in free if covered(r, [rect])))}))
        for rect, text in raw:
            events.append((rect.y0, col_of(rect.x0, rect.x1), "text", (rect, text)))

        for _, _, kind, payload in order(events):
            if kind in ("figure", "table", "equation"):
                items.append({"kind": kind, "page": pno + 1, "y": round(payload["rect"].y0, 1),
                              "caption": payload.get("caption") or payload.get("raw", ""),
                              "rect": box(payload["rect"])})
                continue
            rect, text = payload
            if covered(rect, seeds) or covered(rect, eqs) or CAPTION_RE.search(norm(text)[:40]):
                continue
            label = classify(rect, text)
            if label == "skip":
                continue
            items.append({"kind": label, "page": pno + 1, "col": col_of(rect.x0, rect.x1),
                          "y": round(rect.y0, 1), "text": norm(text)})

    items = stitch(items)
    for i, it in enumerate(items):
        it["i"] = i

    if not args.inspect:
        for it in items:
            if "rect" not in it:
                continue
            name = f"p{it['page']:02d}_{it['kind']}_{it['i']}.png"
            it["img"] = f"assets/{name}"
            doc[it["page"] - 1].get_pixmap(matrix=fitz.Matrix(3, 3),
                                           clip=fitz.Rect(it["rect"])).save(ASSETS / name)

    OUT.mkdir(exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {}
    for it in items:
        stats[it["kind"]] = stats.get(it["kind"], 0) + 1
    print("items:", len(items), stats)
    print("translatable chars:", sum(len(it.get("text", ""))
                                     for it in items if it["kind"] in ("para", "heading", "footnote")))
    if args.inspect:
        print("hyphen decisions (keep | glue):")
        print("  keep:", ", ".join(f"{a}-{b}" for a, b, k in DECISIONS if k))
        print("  glue:", ", ".join(f"{a}{b}" for a, b, k in DECISIONS if not k))
        for it in items:
            print(f"{it['i']:4d} p{it['page']:2d} {it['kind']:8s} "
                  f"{(it.get('text') or it.get('caption') or '')[:86]}")


if __name__ == "__main__":
    sys.exit(main())
