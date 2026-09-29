"""Single-file offline HTML dashboard for the reproduction's progress.

Reads only what is on disk (runs/*/metrics.jsonl, config.yaml, logs/, results/*.json)
and writes one self-contained HTML - inline SVG plus the inline chart script from
mobileie/charts.py, no CDN, no network - so it can be opened from a USB stick or
mailed.  Charts are drawn from the same rows the terminal status uses, so the two
views cannot disagree, and both pages share one renderer (axes, ticks, hover tooltip).

    python scripts/dashboard.py            # write + print the path
    python scripts/dashboard.py --open     # and open it in the default browser
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import argparse
import html
import json
import time
from datetime import datetime, timedelta

from progress import (CARD_COLORS, cfg_of, fig10_card, rate, rows_of,
                      visible_chain)                                    # noqa: E402
from mobileie.charts import (CHART_CSS, CHART_JS, THEME_BOOT_JS, THEME_CSS,
                             THEME_KEY, THEME_PALETTE, chart_for_rows)  # noqa: E402
from mobileie.config import REPRO_ROOT, RESULTS_ROOT                 # noqa: E402

# The colour variables live in mobileie/charts.py so the control board (board.py)
# uses exactly the same theme; only layout rules belong here.
CSS = THEME_CSS + """
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-monospace,SFMono-Regular,Consolas,"Noto Sans Mono CJK SC",monospace;padding:22px}
h1{font-size:19px;margin:0 0 4px}h2{font-size:14px;color:var(--dim);margin:22px 0 8px;
text-transform:uppercase;letter-spacing:.06em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.k{color:var(--dim);font-size:12px}.v{font-size:20px;font-weight:600}
.bar{height:9px;background:var(--line);border-radius:5px;overflow:hidden;margin:7px 0}
.bar>i{display:block;height:100%;background:var(--acc)}
.run>i{background:var(--ok)}.stop>i{background:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
svg{width:100%;height:auto;background:var(--code);border-radius:8px;margin-top:6px}
.tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
 border:1px solid var(--line)}
footer{color:var(--dim);font-size:12px;margin-top:20px}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:10px;
overflow:auto;font-size:12px;white-space:pre-wrap}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:7px;
padding:3px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}
"""

def stage_card(cfg_name, colors=None):
    cfg, run_dir = cfg_of(cfg_name)
    if cfg is None:
        return (f'<div class="card"><span class="k">{html.escape(cfg_name)}</span>'
                f'<div class="bad">配置读不到</div></div>'), None
    o = cfg["optim"]
    stage = o.get("stage", 1)
    warm = int(o.get("warmup_epochs", 0)) if stage == 1 else 0
    planned = int(o["epochs"])
    expected = planned + warm
    rows = rows_of(run_dir)
    done = len(rows)
    r = rate(rows)
    if rows:
        last_ts = datetime.strptime(rows[-1]["ts"], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - last_ts).total_seconds()
        alive = r is not None and age < max(3 * r, 120)
    else:
        age, alive = None, False
    pct = 100.0 * done / expected
    best = max((x for x in rows if "val_psnr" in x), key=lambda x: x["val_psnr"], default=None)
    left = (expected - done) * r if r else None
    finish = (datetime.now() + timedelta(seconds=left)) if left else None
    state = ('<span class="tag ok">运行中</span>' if alive else
             ('<span class="tag">未开始</span>' if not rows else '<span class="tag warn">已停</span>'))
    html_card = f"""<div class="card">
 <div style="display:flex;justify-content:space-between;align-items:center">
  <b>{html.escape(cfg['exp_name'])}</b> {state}</div>
 <div class="k">stage{stage} · {planned} epoch{f' + {warm} warmup' if warm else ''}
   {'· %.1f s/ep' % r if r else ''}</div>
 <div class="bar {('run' if alive else 'stop')}"><i style="width:{pct:.1f}%"></i></div>
 <div><b>{done}/{expected}</b> <span class="k">epoch ({pct:.1f}%)</span></div>
 <div class="k">剩余 {('%.2f h' % (left/3600)) if left else '—'}
   {('→ ' + finish.strftime('%m-%d %H:%M')) if finish else ''}</div>
 <div class="k">best val PSNR <b class="ok">{best['val_psnr']:.3f}</b> @ep{best['epoch']}
   {('· 最新 %.4f' % rows[-1]['val_psnr']) if 'val_psnr' in rows[-1] and rows else ''}</div>
 {chart_for_rows(rows, 'train_loss', title='train_loss（log10 · 越低越好）', log=True,
                palette=THEME_PALETTE, colors=colors)}
 {chart_for_rows(rows, 'val_psnr', title='val_psnr（dB · 越高越好）', unit='dB',
                palette=THEME_PALETTE, colors=colors)}
</div>""" if (best or rows) else f"""<div class="card">
 <div style="display:flex;justify-content:space-between"><b>{html.escape(cfg['exp_name'])}</b>
  {state}</div>
 <div class="k">stage{stage} · {planned} epoch · 未开始</div>
 <div class="bar"><i style="width:0%"></i></div>
</div>"""
    return html_card, (r if r else None)


# FIG10_ARMS / arm_rows / fig10_card 现在放在 progress.py，两个页面共用一份。
def table_card(title, key, metrics_key="metrics"):
    path = RESULTS_ROOT / f"eval_{key}.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text(encoding="utf-8"))
    m, ref = d.get(metrics_key, {}), d.get("reference", {})
    rows = ""
    for label, mk, rk, fmt in (("PSNR", "psnr", "psnr", ".3f"), ("SSIM", "ssim", "ssim", ".4f"),
                               ("LPIPS", "lpips", "lpips", ".4f")):
        if mk not in m:
            continue
        ours = m[mk]
        paper = ref.get(rk)
        if paper is None:
            rows += f"<tr><td>{label}</td><td>—</td><td>{ours:{fmt}}</td><td>—</td></tr>"
            continue
        delta = ours - paper
        cls = "ok" if abs(delta) <= 0.05 else ("warn" if abs(delta) <= 0.5 else "bad")
        rows += (f'<tr><td>{label}</td><td>{paper:{fmt}}</td><td>{ours:{fmt}}</td>'
                 f'<td class="{cls}">{delta:+{fmt}}</td></tr>')
    n = m.get("n_images", "")
    return (f'<div class="card"><b>{html.escape(title)}</b> '
            f'<span class="k">{n} 图</span><table><tr><th>指标</th><th>论文</th>'
            f'<th>复现</th><th>Δ</th></tr>{rows}</table></div>') if rows else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", dest="open_it")
    args = ap.parse_args()

    cards, rates = [], []
    for cfg_name in visible_chain():
        card, r = stage_card(cfg_name, CARD_COLORS.get(cfg_name))
        cards.append(card)
        if r:
            rates.append(r)
    global_rate = rates[0] if rates else None
    gpu = ""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
                              "memory.total,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        name, util, memu, memt, pw = [v.strip() for v in out.splitlines()[0].split(",")]
        gpu = f"{name} · {util}% · 显存 {memu}/{memt} MiB · {pw} W"
    except Exception:                                              # noqa: BLE001
        gpu = "nvidia-smi 不可用"

    tables = (table_card("Table 1 · LOLv1 (LLE)", "pretrained")
              + table_card("Table 2 · UIEB (UIE)", "uieb_native")
              + table_card("Table 3 · ZRR (ISP)", "zrr"))

    fig10 = fig10_card()

    blocks = []
    for lg in sorted((REPRO_ROOT / "logs").glob("*.log")):
        lines = [x for x in lg.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        if lines:
            blocks.append((lg.name, "\n".join(lines[-4:])))
    tail_html = "".join(
        f'<div class="k" style="margin-top:10px">{html.escape(n)}</div>'
        f'<pre>{html.escape(t)}</pre>' for n, t in blocks[-4:]) or '<pre>（无日志）</pre>'

    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MobileIE 复现仪表盘</title>
<script>/* before first paint: light is the default, the last choice is remembered */
{THEME_BOOT_JS}</script>
<style>{CSS}{CHART_CSS}</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:flex-start">
<h1>MobileIE (ICCV 2025) 复现进度</h1>
<button id="thbtn" onclick="ieToggleTheme()" title="浅色/深色切换">主题：浅色</button></div>
<div class="k">{time.strftime('%Y-%m-%d %H:%M:%S')} · {html.escape(gpu)} · 单文件离线，F5 刷新</div>
<h2>训练阶段</h2><div class="grid">{''.join(cards)}</div>
{f'<h2>IWO 与无 IWO 对照</h2><div class="grid">{fig10}</div>' if fig10 else ''}
<h2>作者权重侧三表核对</h2><div class="grid">{tables or '<div class="card">尚无 eval_*.json</div>'}</div>
<h2>最近日志</h2>{tail_html}
<footer>数据源：runs/*&#47;metrics.jsonl、results/eval_*.json、logs/*。
刷新方式：run.cmd dash（重新生成并打开）。</footer>
<script>{CHART_JS}
function ieToggleTheme(){{var t=document.documentElement.getAttribute('data-theme')==='dark'
 ?'light':'dark';document.documentElement.setAttribute('data-theme',t);
 try{{localStorage.setItem('{THEME_KEY}',t)}}catch(e){{}}ieSyncTheme()}}
function ieSyncTheme(){{var b=document.getElementById('thbtn');if(b)b.textContent=
 '主题：'+(document.documentElement.getAttribute('data-theme')==='dark'?'深色':'浅色')}}
ieSyncTheme();ieRenderCharts();</script></body></html>"""

    out = RESULTS_ROOT / "dashboard.html"
    out.write_text(doc, encoding="utf-8")
    print(out)
    if args.open_it:
        import os
        os.startfile(str(out)) if hasattr(os, "startfile") else \
            subprocess_run_fallback(str(out))


def subprocess_run_fallback(path):
    import subprocess
    subprocess.run(["xdg-open", path], check=False)


if __name__ == "__main__":
    main()
