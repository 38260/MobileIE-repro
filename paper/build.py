"""Build the bilingual (EN paragraph + ZH paragraph) HTML from the extracted PDF.

    python paper/build.py        # -> paper/bilingual/bilingual.html

Layout keeps the paper's own order: title, abstract, sections, floats and equations
in place, references at the end. Each text block is printed in English first and then
in Chinese; figures and tables are the page crops, so their printed captions and rules
come through unchanged (only a Chinese caption is added underneath).
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bilingual"
sys.path.insert(0, str(OUT))

from zh import CAPTIONS, ZH  # noqa: E402

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
:root { --zh: #8a4b1f; --rule: #e3cfae; --ink: #1c1c1c; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); font-size: 10.5pt; line-height: 1.55;
       -webkit-print-color-adjust: exact; }
.en, .zh { text-align: justify; hyphens: auto; }
.en { font-family: Georgia, 'Times New Roman', serif; }
.zh { font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; color: var(--zh);
      border-left: 3px solid var(--rule); padding-left: 9pt; margin-top: 4pt; }
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; line-height: 1.3;
             margin: 16pt 0 2pt; break-after: avoid; }
h1 { font-size: 17pt; text-align: center; margin-top: 0; }
h1.zh { border: none; padding: 0; text-align: center; font-size: 16pt; margin-top: 2pt; }
h2 { font-size: 12.5pt; }
h3 { font-size: 11pt; }
h2.zh, h3.zh { border: none; padding: 0; margin: 0 0 8pt; font-size: inherit; }
.authors { text-align: center; font-size: 10pt; margin-top: 6pt; }
.authors.zh { border: none; padding: 0; text-align: center; margin: 1pt 0 0; }
figure { margin: 10pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figure.eq img { max-height: 34pt; }
figcaption { font-family: 'Microsoft YaHei', sans-serif; font-size: 8.6pt; color: var(--zh);
             text-align: left; margin-top: 3pt; line-height: 1.45; }
figcaption b { color: #b06a2c; }
.foot { font-size: 9pt; }
.refs { font-size: 8.2pt; line-height: 1.42; text-align: justify;
        font-family: Georgia, 'Times New Roman', serif; }
.note { font-family: 'Microsoft YaHei', sans-serif; font-size: 8.6pt; color: #6b5a45;
        border-top: 1px solid var(--rule); padding-top: 4pt; margin: 4pt 0 10pt; }
"""


def zh_blocks(text: str, tag: str, cls: str = "") -> list[str]:
    """One block per Chinese paragraph, so a stitched English paragraph still reads as
    the two translations it came from."""
    klass = f"{cls} zh" if cls else "zh"
    return [f'<{tag} class="{klass}">{esc(part)}</{tag}>'
            for part in text.split("\n\n") if part]


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def main():
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    ref_at = min(i["i"] for i in items if i.get("text") == "References")
    body = [i for i in items if i["i"] < ref_at and i.get("kind") in ("para", "heading", "footnote")]

    def key(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    used = [False] * len(ZH)
    zh_by_index = {}
    for block in body:
        text = key(block["text"])
        hits = [n for n, (prefix, _) in enumerate(ZH) if not used[n] and key(prefix) in text]
        if not hits:
            raise SystemExit(f"no translation for block {block['i']}: {block['text'][:70]}")
        for n in hits:
            used[n] = True
        zh_by_index[block["i"]] = "\n\n".join(ZH[n][1] for n in hits)
    left = [n for n, seen in enumerate(used) if not seen]
    if left:
        raise SystemExit(f"{len(left)} translations unused, first: {ZH[left[0]][0]}")
    out = []
    for it in items:
        kind, i = it["kind"], it["i"]
        if kind in ("figure", "table"):
            label = zh_caption(kind, it["caption"])
            out.append(f'<figure><img src="{it["img"]}" alt="{esc(it["caption"][:60])}">'
                       f'<figcaption><b>{esc(kind_label(kind, it["caption"]))}</b>　{esc(label)}</figcaption></figure>')
        elif kind == "equation":
            out.append(f'<figure class="eq"><img src="{it["img"]}" alt="equation"></figure>')
        elif kind == "heading":
            tag = "h1" if i == 0 else ("h2" if re.match(r"^\d+\.\s", it["text"]) else "h3")
            out.append(f'<{tag} class="en">{esc(it["text"])}</{tag}>')
            extra = {"References": "参考文献"}.get(it["text"], "")
            out.extend(zh_blocks(zh_by_index.get(i, "") or extra, tag))
        elif kind == "footnote":
            out.append(f'<p class="foot en">{esc(it["text"])}</p>')
            out.extend(zh_blocks(zh_by_index.get(i, ""), "p", "foot"))
        elif kind == "para" and i >= ref_at:
            out.append(f'<p class="refs">{esc(it["text"])}</p>')
        elif kind == "para":
            cls = "authors" if i in (1, 2) else "para"
            if i == 0:
                out.append(f'<h1 class="en">{esc(it["text"])}</h1>')
                out.extend(zh_blocks(zh_by_index.get(i, ""), "h1"))
                continue
            out.append(f'<p class="{cls} en">{esc(it["text"])}</p>')
            if i == 1:
                continue            # author names carry no translation; the EN line is enough
            out.extend(zh_blocks(zh_by_index.get(i, ""), "p", cls))

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>MobileIE (ICCV 2025) 中英对照</title><style>{CSS}</style></head>
<body>
<p class="note">MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image
Enhancement on Mobile Devices — ICCV 2025 中英对照版。英文为原文逐段照录，中文为紧随其后的译文；
图、表与公式按原书版式整块裁入，故其英文图注已包含在图片内，图下只补中文说明。参考文献保留英文原文。</p>
{chr(10).join(out)}
</body></html>
"""
    target = OUT / "bilingual.html"
    target.write_text(html_doc, encoding="utf-8")
    print(f"wrote {target} ({len(html_doc) // 1024} KB, {len(items)} blocks)")


def float_key(kind: str, caption: str) -> str:
    match = re.match(r"(?:Figure|Table)\s+(\d+)\.", caption)
    if not match:
        raise SystemExit(f"caption without a number: {caption[:60]}")
    return f"{kind} {match.group(1)}"


def kind_label(kind: str, caption: str) -> str:
    return caption.split(".", 1)[0] + "（中文）"


def zh_caption(kind: str, caption: str) -> str:
    key = float_key(kind, caption)
    if key not in CAPTIONS:
        raise SystemExit(f"untranslated caption: {key}")
    return CAPTIONS[key]

if __name__ == "__main__":
    main()
