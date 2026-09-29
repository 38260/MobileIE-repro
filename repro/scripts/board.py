"""Local control board: trigger training from the browser and watch it live.

Stdlib only (http.server + subprocess), bound to 127.0.0.1 - it can start and kill
processes, so it must never listen on a reachable interface.

    python scripts/board.py [--port 8765] [--open]
    run.cmd board

The page polls /api/status every 3 s; every number comes from the same files the
terminal status reads (runs/*/metrics.jsonl, logs/, results/eval_*.json), so the
board, run.cmd status and report.py can never disagree.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from progress import (CARD_COLORS, FIG10_ARMS, cfg_of, rate, rows_of,
                      visible_chain)                                   # noqa: E402
from mobileie.charts import (CHART_CSS, CHART_JS, EPOCH_TICK_STEP, THEME_BOOT_JS,
                             THEME_CSS, THEME_KEY, THEME_PALETTE)          # noqa: E402
from mobileie.config import REPRO_ROOT, RESULTS_ROOT                 # noqa: E402

IS_WIN = os.name == "nt"
JOBS = {}          # name -> {"started": ts, "rc": int|None, "log": str}


# --------------------------------------------------------------------------- actions
def live_chain_pids():
    """PIDs of a running chain, however it was launched (board, run.cmd, by hand)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
          "'^(python\\.exe|cmd\\.exe)$' -and $_.CommandLine -match 'run_two_stage|train\\.py' } "
          "| ForEach-Object { $_.ProcessId }")
    cmd = (["powershell", "-NoProfile", "-Command", ps] if IS_WIN else
           ["sh", "-c", "pgrep -f 'run_two_stage|train\\.py' | tr '\\n' ' '"])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:                                          # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                              "memory.total,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        util, used, total, power, temp = [v.strip() for v in out.splitlines()[0].split(",")]
        return {"util": int(util), "mem_used": int(used), "mem_total": int(total),
                "power": power, "temp": temp}
    except Exception:                                          # noqa: BLE001
        return {}


def spawn_detached(args, logfile):
    """Start a chain so it outlives the board process (and the browser tab)."""
    log = open(REPRO_ROOT / "logs" / logfile, "a", encoding="utf-8")
    kwargs = {"cwd": str(REPRO_ROOT), "stdout": log, "stderr": subprocess.STDOUT,
              "stdin": subprocess.DEVNULL}
    if IS_WIN:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.DETACHED_PROCESS)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs)


def spawn_in_console(args):
    """Open a VISIBLE console window running ``args`` so the operator can watch it.

    CREATE_NEW_CONSOLE gives the child its own window (with Windows Terminal as the
    default terminal app that shows up as a new tab/window).  CREATE_BREAKAWAY_FROM_JOB
    keeps the run alive after the board is closed; it is dropped when the surrounding
    job forbids breakaway, which raises OSError.

    The launcher paths contain no spaces, so cmd's quoting rules are not exercised.
    """
    flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        return subprocess.Popen(args, cwd=str(REPRO_ROOT),
                                creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        return subprocess.Popen(args, cwd=str(REPRO_ROOT), creationflags=flags)


def run_job(name, args):
    def body():
        JOBS[name] = {"started": time.strftime("%H:%M:%S"), "rc": None,
                      "log": f"logs/job_{name}.log"}
        with open(REPRO_ROOT / "logs" / f"job_{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(args, cwd=str(REPRO_ROOT), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        JOBS[name]["rc"] = rc
        JOBS[name]["ended"] = time.strftime("%H:%M:%S")
    threading.Thread(target=body, daemon=True).start()


def do_action(action, payload):
    runner = "scripts\\run_two_stage.cmd" if IS_WIN else "scripts/run_two_stage.sh"
    py = str(REPRO_ROOT.parent / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python"))
    if action == "start":
        if live_chain_pids():
            return False, "已有训练链路在跑，先点停止"
        chain = payload.get("chain") or ["lolv1_stage1", "lolv1_stage2_iwo"]
        if IS_WIN:
            # own console window: the chain prints its per-epoch lines there, and
            # run_two_stage.cmd's `timeout` backoff needs a real console to work
            spawn_in_console(["cmd", "/k", runner] + chain)
        else:
            spawn_detached(["bash", runner] + chain, "board_launch.log")
        return True, f"已在新终端窗口启动链路: {' -> '.join(chain)}"
    if action == "start_ctrl":
        # Phase-1 control arm: a single config, 2000 epochs, no IWO.  train.py finds
        # runs/lolv1_ctrl_noiwo/ckpt/state_last.pt by itself, so clicking this always
        # resumes instead of restarting from scratch.  It is launched in its own
        # console window so the per-epoch lines stay visible while it runs.
        if live_chain_pids():
            return False, "已有训练在跑，先点停止"
        launcher = REPRO_ROOT.parent / "train_ctrl.bat"
        if IS_WIN and launcher.exists():
            spawn_in_console(["cmd", "/k", str(launcher)])
        else:
            spawn_detached([py, "scripts/train.py", "--config", "lolv1_ctrl_noiwo"],
                           "ctrl_train.log")
        return True, "已在新终端窗口启动对照组（2000 ep · 不加 IWO · 逐轮打印）"
    if action == "smoke":
        if live_chain_pids():
            return False, "已有训练在跑，先停止再跑冒烟"
        run_job("smoke", [py, "scripts/train.py", "--config", "lolv1_smoke",
                          "--run-name", "board_smoke"])
        return True, "冒烟测试已启动（2 epoch，约 1 分钟，产物 runs/board_smoke）"
    if action == "stop":
        pids = live_chain_pids()
        if not pids:
            return True, "没有正在运行的训练"
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
              "'^(python\\.exe|cmd\\.exe)$' -and $_.CommandLine -match 'run_two_stage|train\\.py' } "
              "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        kill = (["powershell", "-NoProfile", "-Command", ps] if IS_WIN else
                ["pkill", "-f", "run_two_stage|train.py"])
        subprocess.run(kill, capture_output=True, timeout=30)
        return True, f"已停止 {len(pids)} 个进程（state_last.pt 保留，可继续）"
    if action == "eval":
        run_job("eval", [py, "scripts/evaluate.py", "--config", "lolv1_stage1",
                         "--run-name", "lolv1_stage1", "--tag", "stage1"])
        return True, "评估已在后台运行，完成后写 results/eval_stage1.json"
    if action == "report":
        run_job("report", [py, "scripts/report.py"])
        return True, "报告重生成中"
    if action == "verify":
        run_job("verify", [py, "scripts/evaluate.py", "--config", "lolv1_pretrained",
                           "--tag", "pretrained"])
        return True, "官方权重核对中（Table 1）"
    return False, f"未知动作 {action}"


# --------------------------------------------------------------------------- status
def stage_status(cfg_name):
    cfg, run_dir = cfg_of(cfg_name)
    if cfg is None:
        return {"config": cfg_name, "missing": True}
    o = cfg["optim"]
    stage = o.get("stage", 1)
    warm = int(o.get("warmup_epochs", 0)) if stage == 1 else 0
    planned = int(o["epochs"])
    rows = rows_of(run_dir)
    r = rate(rows)
    done = len(rows)
    expected = planned + warm
    best = max((x for x in rows if "val_psnr" in x), key=lambda x: x["val_psnr"], default=None)
    left = (expected - done) * r if (r and done < expected) else None
    age = None
    if rows:
        age = (datetime.now() - datetime.strptime(rows[-1]["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    return {"config": cfg_name, "exp": cfg["exp_name"], "run_dir": run_dir.name,
            "stage": stage, "batch": o.get("batch_size"),
            "epochs": planned, "warmup": warm, "done": done, "expected": expected,
            "s_per_epoch": r, "left_s": left, "rows": rows,
            "best": ({"psnr": best["val_psnr"], "epoch": best["epoch"]} if best else None),
            "last": rows[-1] if rows else None, "age_s": age,
            "alive": bool(r and age is not None and age < max(3 * r, 120))}


def board_status():
    pids = live_chain_pids()
    stages = []
    for c in visible_chain():                # 已并入别的卡的阶段不再单独出卡
        st = stage_status(c)
        st["colors"] = CARD_COLORS.get(c, {})
        stages.append(st)
    total_left = sum(s.get("left_s") or 0 for s in stages if not s.get("missing"))
    if not any(s.get("rows") for s in stages):
        total_left = None
    evals = {}
    for f in RESULTS_ROOT.glob("eval_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            evals[f.stem[5:]] = {"metrics": d.get("metrics", {}), "reference": d.get("reference", {})}
        except Exception:                                      # noqa: BLE001
            continue
    logs = {}
    for lg in sorted((REPRO_ROOT / "logs").glob("*.log")):
        lines = [x for x in lg.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        if lines:
            logs[lg.name] = lines[-12:]
    return {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "chain_pids": pids, "training": bool(pids),
            "gpu": gpu_stats(), "stages": stages, "total_left_s": total_left,
            "fig10_arms": [{"config": c, "offset": o, "color": col, "label": lab}
                           for c, o, col, lab in FIG10_ARMS],
            "evals": evals, "jobs": dict(JOBS), "logs": logs,
            "eta_finish": (datetime.now().timestamp() + total_left) if total_left else None}


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MobileIE 复现控制台</title><style>
__THEME_CSS__
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);padding:18px;
font:14px/1.5 ui-monospace,Consolas,"Noto Sans Mono CJK SC",monospace}
h1{font-size:18px;margin:0 0 2px}h2{font-size:12px;color:var(--dim);letter-spacing:.07em;
text-transform:uppercase;margin:20px 0 8px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.k{color:var(--dim);font-size:12px}.big{font-size:19px;font-weight:600}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:8px 13px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}button.go{border-color:var(--ok)}button.stop{border-color:var(--bad)}
button:disabled{opacity:.45;cursor:not-allowed}
.bar{height:9px;background:var(--line);border-radius:5px;overflow:hidden;margin:6px 0}
.bar>i{display:block;height:100%;background:var(--acc);transition:width .6s}
.on .bar>i{background:var(--ok)}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
svg{width:100%;height:auto;background:var(--code);border-radius:8px;margin-top:6px}
__CHART_CSS__
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:9px;font-size:12px;
max-height:190px;overflow:auto;margin:4px 0 0}
.tag{padding:1px 8px;border-radius:20px;font-size:11px;border:1px solid var(--line)}
#toast{position:fixed;right:16px;bottom:16px;background:var(--card);border:1px solid var(--acc);
padding:10px 14px;border-radius:8px;opacity:0;transition:opacity .3s;max-width:60vw}
</style><script>__THEME_BOOT_JS__</script></head><body>
<div class="row" style="justify-content:space-between;align-items:baseline">
<h1>MobileIE 复现控制台</h1>
<button id="thbtn" onclick="ieToggleTheme()" title="浅色/深色切换">主题：浅色</button></div>
<div class="k" id="top">连接中…</div>
<h2>控制</h2><div class="card"><div class="row">
<button class="go" onclick="act('start')">▶ 启动两阶段训练</button>
<button class="go" onclick="act('start_ctrl')">▶ 启动对照组（2000 ep · 不加 IWO）</button>
<button onclick="act('smoke')">⚗ 冒烟测试（2 ep，约 1 分钟）</button>
<button class="stop" onclick="act('stop')">■ 停止（保留断点）</button>
<button onclick="act('verify')">核对官方权重</button>
<button onclick="act('eval')">评估 stage1</button>
<button onclick="act('report')">重生成报告</button>
</div><div class="k" style="margin-top:8px">「启动两阶段训练」= stage1 1000ep → IWO stage2 1000ep，本机约 17.8 h；
「启动对照组」= 单条 2000ep、不加 IWO，点击即从当前断点续跑。
两者都会弹出一个<b>新的终端窗口</b>，在里面逐轮打印进度——那个窗口别关（关了训练就停）。
停止：在终端里按 Ctrl+C，或点这里的「停止」。state_last.pt 每轮保存，重新启动即续跑。仅监听 127.0.0.1。</div></div>
<h2>训练阶段</h2><div class="grid" id="stages"></div>
<h2>IWO 与无 IWO 对照</h2><div class="grid" id="fig10"></div>
<h2>作者权重三表核对</h2><div class="grid" id="tables"></div>
<h2>日志</h2><div class="grid" id="logs"></div>
<div id="toast"></div>
<script>
__CHART_JS__
let busy=false;
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.style.opacity=1;
 setTimeout(()=>t.style.opacity=0,4200)}
function ieSyncTheme(){const b=document.getElementById('thbtn');
 if(b)b.textContent='主题：'+(document.documentElement.getAttribute('data-theme')==='dark'?'深色':'浅色')}
function ieToggleTheme(){const t=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';
 document.documentElement.setAttribute('data-theme',t);
 try{localStorage.setItem('__THEME_KEY__',t)}catch(e){}ieSyncTheme()}
ieSyncTheme();
function fmtH(s){if(s==null)return '—';const h=s/3600;return h>=1?h.toFixed(2)+' h':Math.round(s)+' s'}
function clock(ts){if(!ts)return '';const d=new Date(ts*1000);
 return d.toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})}
function ieDiv(spec){return '<div class="iechart" data-c="'+encodeURIComponent(JSON.stringify(spec))+
 '" style="min-height:'+((spec.h||210)-40)+'px"></div>'}
function chart(rows,key,logc,title,unit,colors){const g={};
 rows.forEach(r=>{if(r[key]==null)return;const p=r.phase||'train';(g[p]=g[p]||[]).push(
  [r.epoch,r[key],{lr:r.lr,ts:r.ts}])});
 const pal=__THEME_PALETTE__;
 const series=Object.entries(g).map(([p,pts],i)=>({label:p,color:(colors||{})[p]||pal[i%pal.length],pts}));
 return ieDiv({title:title||key,unit:unit||'',log:!!logc,h:210,xstep:__EPOCH_STEP__,series:series})}
function fig10(stages,arms){
 if(!arms||!arms.length)return '';
 const by={};stages.forEach(s=>{by[s.config]=s});
 let live=0,stats='',sl=[],sv=[];
 arms.forEach(a=>{const st=by[a.config],rows=((st&&st.rows)||[]).filter(r=>r.phase!=='warmup');
  const mk=k=>rows.filter(r=>r[k]!=null).map(r=>[r.epoch+a.offset,r[k],{lr:r.lr,ts:r.ts}]);
  sl.push({label:a.label,color:a.color,pts:mk('train_loss')});
  sv.push({label:a.label,color:a.color,pts:mk('val_psnr')});
  if(rows.length>=2){live++;
   const lo=rows[0].epoch+a.offset,hi=rows[rows.length-1].epoch+a.offset;
   const b=rows.reduce((x,y)=>((y.val_psnr||-1e9)>(x.val_psnr||-1e9)?y:x));
   stats+='<div><span style="color:'+a.color+'">■</span> <b>'+esc(a.label)+'</b> <span class="k">'+
    esc(a.config)+' · 全局 ep'+lo+'–'+hi+' · best val '+(b.val_psnr||0).toFixed(4)+
    ' @ep'+(b.epoch+a.offset)+'</span></div>'}});
 if(!live)return '';
 return '<div class="card"><b>两臂对照 · 论文 Fig. 10 口径</b>'+stats+
  '<div class="k">蓝 = 无 IWO 的连续 2000 轮；红 = 第 1000 轮后接入 IWO（阶段二 epoch 平移到 1001–2000）。已排除 10 轮 warmup。'+
  '▲/▼ = 每条曲线的最高/最低点（显示「数值@epoch」）；横轴每格 '+__EPOCH_STEP__+' 轮。</div>'+
  ieDiv({title:'train_loss（log10 · 越低越好）',unit:'',log:true,h:240,xstep:__EPOCH_STEP__,series:sl})+
  ieDiv({title:'val_psnr（dB · 越高越好）',unit:'dB',log:false,h:240,xstep:__EPOCH_STEP__,series:sv})+
  '</div>'}
function render(s){
 document.getElementById('top').innerHTML=s.ts+' · '+(s.training?
  '<b class="ok">训练中</b>（'+s.chain_pids.length+' 进程）':'<span class="warn">空闲</span>')+
  ' · GPU '+(s.gpu.util!=null?s.gpu.util+'% / '+(s.gpu.mem_used/1024).toFixed(1)+'/'+
  (s.gpu.mem_total/1024).toFixed(0)+' GB / '+s.gpu.power+' W / '+s.gpu.temp+'°C':'n/a')+
  ' · 合计剩余 <b>'+fmtH(s.total_left_s)+'</b>'+(s.eta_finish?' → '+clock(s.eta_finish):'');
 // re-drawing while the pointer is inside a chart would kill the tooltip every 3 s
 if(!document.querySelector('.iechart:hover')){
 document.getElementById('stages').innerHTML=s.stages.map(st=>{
  if(st.missing)return '<div class="card"><b>'+esc(st.config)+'</b><div class="bad">配置缺失</div></div>';
  const pct=st.expected?100*st.done/st.expected:0;
  const rows=st.rows||[];
  return '<div class="card'+(st.alive?' on':'')+'"><div style="display:flex;justify-content:space-between">'+
   '<b>'+esc(st.exp)+'</b><span class="tag">'+(st.alive?'<b class="ok">运行中</b>':
   (rows.length?'<span class="warn">已停</span>':'未开始'))+'</span></div>'+
   '<div class="k">stage'+st.stage+' · '+st.epochs+' ep'+(st.warmup?' + '+st.warmup+' warmup':'')+
   ' · bs='+st.batch+(st.s_per_epoch?' · '+st.s_per_epoch.toFixed(1)+' s/ep':'')+
   (st.run_dir!==st.exp?' · <span class="bad">目录 '+esc(st.run_dir)+'</span>':'')+'</div>'+
   '<div class="bar"><i style="width:'+pct.toFixed(1)+'%"></i></div>'+
   '<div><b>'+st.done+'/'+st.expected+'</b> <span class="k">epoch ('+pct.toFixed(1)+'%)</span>'+
   ' · 剩 '+fmtH(st.left_s)+(st.left_s?' → '+clock(Date.now()/1000+st.left_s):'')+'</div>'+
   (st.best?'<div class="k">best val PSNR <b class="ok">'+st.best.psnr.toFixed(3)+'</b> @ep'+
    st.best.epoch+'</div>':'')+
   chart(rows,'train_loss',1,'train_loss（log10 · 越低越好）','',st.colors)+
   chart(rows,'val_psnr',0,'val_psnr（dB · 越高越好）','dB',st.colors)+'</div>'}).join('');
 document.getElementById('fig10').innerHTML=fig10(s.stages,s.fig10_arms)}
 const T=[['pretrained','Table 1 · LOLv1'],['uieb_native','Table 2 · UIEB'],['zrr','Table 3 · ZRR']];
 document.getElementById('tables').innerHTML=T.map(([k,ti])=>{const e=s.evals[k];
  if(!e)return '<div class="card"><b>'+ti+'</b><div class="k">未评估</div></div>';
  const r=e.reference||{},m=e.metrics||{};let rows='';
  for(const [lb,key,f] of [['PSNR','psnr',3],['SSIM','ssim',4],['LPIPS','lpips',4]]){
   if(m[key]==null)continue;const p=r[key];
   let d='<span class="k">—</span>';
   if(p!=null){const dv=m[key]-p;const c=Math.abs(dv)<=0.05?'ok':(Math.abs(dv)<=0.5?'warn':'bad');
    d='<span class="'+c+'">'+(dv>=0?'+':'')+dv.toFixed(f)+'</span>'}
   rows+='<tr><td>'+lb+'</td><td>'+(p!=null?p.toFixed(f):'—')+'</td><td>'+m[key].toFixed(f)+'</td><td>'+d+'</td></tr>'}
  return '<div class="card"><b>'+ti+'</b> <span class="k">'+(m.n_images||'')+' 图</span>'+
   '<table><tr><th>指标</th><th>论文</th><th>复现</th><th>Δ</th></tr>'+rows+'</table></div>'}).join('');
 document.getElementById('logs').innerHTML=Object.entries(s.logs).slice(-6).map(([n,l])=>
  '<div class="card"><span class="k">'+esc(n)+'</span><pre>'+esc(l.join('\\n'))+'</pre></div>').join('');
 ieRenderCharts();
}
async function poll(){if(busy)return;busy=true;
 try{const r=await fetch('/api/status');render(await r.json())}catch(e){toast('看板服务断开: '+e)}
 finally{busy=false}}
async function act(a){
 if(a==='start'&&!confirm('启动两阶段训练？本机约 17.8 小时，期间显卡持续满载。'))return;
 if(a==='start_ctrl'&&!confirm('启动对照组？2000 轮、不加 IWO，会从当前断点自动续跑。'))return;
 const body={action:a,chain:['lolv1_stage1','lolv1_stage2_iwo']};
 const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body)});
 const j=await r.json();toast((j.ok?'✓ ':'✗ ')+j.msg);poll()}
poll();setInterval(poll,3000);
</script></body></html>"""

PAGE = (PAGE
        .replace("__THEME_CSS__", THEME_CSS)
        .replace("__THEME_BOOT_JS__", THEME_BOOT_JS)
        .replace("__THEME_KEY__", THEME_KEY)
        .replace("__THEME_PALETTE__", json.dumps(list(THEME_PALETTE)))
        .replace("__EPOCH_STEP__", str(EPOCH_TICK_STEP))
        .replace("__CHART_CSS__", CHART_CSS)
        .replace("__CHART_JS__", CHART_JS))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            self._send(200, json.dumps(board_status(), ensure_ascii=False))
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if not self.path.startswith("/api/action"):
            return self._send(404, '{"ok":false,"msg":"not found"}')
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            ok, msg = do_action(payload.get("action", ""), payload)
        except Exception as exc:                               # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        self._send(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    (REPRO_ROOT / "logs").mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"MobileIE control board: {url}   (Ctrl+C to stop the board; training keeps running)")
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        # a second `run.cmd board` is the common case: point at the live one instead
        print(f"port {args.port} busy ({exc}); a board is probably already running -> open {url}")
        if args.open:
            webbrowser.open(url)
        return
    if args.open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
