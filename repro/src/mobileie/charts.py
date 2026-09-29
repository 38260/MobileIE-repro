"""One interactive line-chart renderer shared by the dashboard and the control board.

Both pages used to hand-roll axis-less polylines, which is why they drifted apart.
Charts are described by a small JSON spec (URI-encoded into a data attribute) and
drawn by CHART_JS: epoch on x, metric on y, grid + ticks, an optional best-point
marker, and a hover tooltip that reports the exact epoch, value, lr and timestamp.

    spec = {"title": ..., "unit": ..., "log": bool, "h": px,
            "series": [{"label": "train_s1", "color": "#4ade80",
                        "pts": [[epoch, value, {"lr": .., "ts": ..}], ...]}],
            "xstep": 250}
"""
from __future__ import annotations

import json
from urllib.parse import quote

PALETTE = ("#60a5fa", "#4ade80", "#fbbf24", "#f472b6", "#a78bfa")

# Every chart in this project puts epoch on x, so they all share one tick grid:
# 250 epochs per division.  The renderer falls back to auto ticks when a range is
# too short to fit two divisions (e.g. the 2-epoch smoke run), so passing this
# unconditionally is safe.  Single source of truth -- board.py injects it into the
# page JS, progress.py/chart_for_rows put it in the spec.
EPOCH_TICK_STEP = 250

# --------------------------------------------------------------------------- theme
# Single source of truth for both pages (dashboard.py and board.py).  Light is the
# default; the dark block is the palette the control board used to hard-code.
# Every chart colour is also written as var(--ie-*, fallback) so a spec colour can
# name a theme variable -- only a CSS declaration may carry var().
THEME_KEY = "ie-dash-theme"

THEME_CSS = """
:root{color-scheme:light;--bg:#fff;--card:#f7f8fa;--fg:#1f2328;--dim:#6b7688;--line:#e2e6ec;
--ok:#15803d;--warn:#b45309;--bad:#b91c1c;--acc:#2563eb;--code:#f6f8fa;
--ie-bg:#fdfdfe;--ie-grid:#e6eaf1;--ie-axis:#b9c0cc;--ie-tick:#6b7688;--ie-label:#8d97a6;
--ie-title:#1f2328;--ie-guide:#9aa5b4;--ie-best:#b45309;
--ie-tipbg:#fff;--ie-tipbd:#c9d3e0;--ie-tipfg:#1f2328;--ie-tipb:#000;
--ie-c1:#2563eb;--ie-c2:#059669;--ie-c3:#d97706;--ie-c4:#db2777;--ie-c5:#7c3aed}
html[data-theme=dark]{color-scheme:dark;--bg:#12151a;--card:#191d24;--fg:#e6e9ef;--dim:#8b95a7;
--line:#232936;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--acc:#60a5fa;--code:#0d1014;
--ie-bg:#0c0f13;--ie-grid:#1e2734;--ie-axis:#3a4557;--ie-tick:#8b95a7;--ie-label:#6b7688;
--ie-title:#c9d3e0;--ie-guide:#5b6b80;--ie-best:#fbbf24;
--ie-tipbg:#0f1720;--ie-tipbd:#3b556e;--ie-tipfg:#dbe4ef;--ie-tipb:#fff;
--ie-c1:#60a5fa;--ie-c2:#4ade80;--ie-c3:#fbbf24;--ie-c4:#f472b6;--ie-c5:#a78bfa}
"""

# Applied before the body paints, so a stored dark choice does not flash light first.
THEME_BOOT_JS = ("(function(){var t='light';try{t=localStorage.getItem('" + THEME_KEY +
                 "')||'light'}catch(e){}document.documentElement.setAttribute('data-theme',t)})();")

# Line colours resolved against the active theme instead of frozen hex.
THEME_PALETTE = tuple(f"var(--ie-c{i},{c})" for i, c in enumerate(PALETTE, 1))

CHART_CSS = """
.iechart{position:relative;background:var(--ie-bg,#0c0f13);border-radius:8px;margin-top:8px}
.iechart svg{display:block;width:100%;height:auto;background:none;border-radius:8px;margin:0}
.iegrid{stroke:var(--ie-grid,#1e2734);stroke-width:1}
.ieaxis{stroke:var(--ie-axis,#3a4557);stroke-width:1}
.ieline{fill:none;stroke-width:1.7}
.ietick{fill:var(--ie-tick,#8b95a7);font-size:10px;font-family:ui-monospace,Menlo,Consolas,monospace}
.ielabel{fill:var(--ie-label,#6b7688);font-size:10px}
.ietitle{fill:var(--ie-title,#c9d3e0);font-size:11px;font-family:ui-monospace,Menlo,Consolas,monospace}
.ielegend{fill:var(--ie-tick,#8b95a7);font-size:10px}
.iebest{fill:var(--ie-best,#fbbf24)}
.ieguide{stroke:var(--ie-guide,#5b6b80);stroke-width:1;stroke-dasharray:3 3}
.ietip{position:absolute;pointer-events:none;background:var(--ie-tipbg,#0f1720);
 border:1px solid var(--ie-tipbd,#3b556e);
 border-radius:6px;padding:6px 9px;font-size:11px;line-height:1.55;color:var(--ie-tipfg,#dbe4ef);
 font-family:ui-monospace,Menlo,Consolas,monospace;white-space:nowrap;opacity:0;
 transition:opacity .12s;z-index:20}
.ietip b{color:var(--ie-tipb,#fff)}
"""

CHART_JS = r"""
function ieNum(v){if(v==null||isNaN(v))return '—';var a=Math.abs(v);
  return a>=1000?v.toFixed(0):a>=10?v.toFixed(2):a>=0.01?v.toFixed(4):v.toExponential(2);}
function ieTicks(lo,hi,n){var span=(hi-lo)||1,raw=span/Math.max(n,1),
  mag=Math.pow(10,Math.floor(Math.log10(raw))),nm=raw/mag,
  step=(nm<1.5?1:nm<3?2:nm<7?5:10)*mag,out=[],t=Math.ceil(lo/step)*step;
  for(;t<=hi+1e-9;t+=step)out.push(t);return out;}
function ieXticks(lo,hi,step,iw){var out=[];
  if(step>0){var t=Math.ceil(lo/step)*step;for(;t<=hi+1e-9;t+=step)out.push(t);}
  return out.length>=2?out:ieTicks(lo,hi,Math.max(2,Math.min(7,Math.round(iw/85))));}
function ieDraw(el){
  var spec;try{spec=JSON.parse(decodeURIComponent(el.dataset.c||''));}catch(e){return;}
  var W=Math.max(300,el.clientWidth||560),H=spec.h||210,
      m={t:26,r:16,b:30,l:56},iw=W-m.l-m.r,ih=H-m.t-m.b;
  var all=[];spec.series.forEach(function(s){s.pts.forEach(function(p){all.push(p);});});
  if(all.length<2){el.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+
      '"><text x="14" y="'+(m.t+12)+'" class="ietitle">'+(spec.title||'')+
      '</text><text x="14" y="'+(m.t+30)+'" class="ietick">数据不足（至少 2 个 epoch）</text></svg>';return;}
  var lg=!!spec.log,Y=function(p){return lg?Math.log10(Math.max(p[1],1e-9)):p[1];};
  var xlo=Math.min.apply(null,all.map(function(p){return p[0];})),
      xhi=Math.max.apply(null,all.map(function(p){return p[0];})),
      ylo=Math.min.apply(null,all.map(Y)),yhi=Math.max.apply(null,all.map(Y));
  if(yhi-ylo<1e-12){ylo-=0.5;yhi+=0.5;}
  var pad=(yhi-ylo)*0.08;ylo-=pad;yhi+=pad;
  var X=function(x){return m.l+(x-xlo)/((xhi-xlo)||1)*iw;},
      Yp=function(y){return m.t+ih-(y-ylo)/((yhi-ylo)||1)*ih;};
  var g=[];
  ieTicks(ylo,yhi,4).forEach(function(t){var y=Yp(t);
    g.push('<line class="iegrid" x1="'+m.l+'" y1="'+y.toFixed(1)+'" x2="'+(W-m.r)+'" y2="'+y.toFixed(1)+'"/>');
    g.push('<text class="ietick" x="'+(m.l-7)+'" y="'+(y+3.5).toFixed(1)+'" text-anchor="end">'+
           (lg?ieNum(Math.pow(10,t)):ieNum(t))+'</text>');});
  ieXticks(xlo,xhi,spec.xstep,iw).forEach(function(t){var x=X(t);
    g.push('<line class="iegrid" x1="'+x.toFixed(1)+'" y1="'+m.t+'" x2="'+x.toFixed(1)+'" y2="'+(m.t+ih)+'"/>');
    g.push('<text class="ietick" x="'+x.toFixed(1)+'" y="'+(m.t+ih+15)+'" text-anchor="middle">'+Math.round(t)+'</text>');});
  g.push('<line class="ieaxis" x1="'+m.l+'" y1="'+(m.t+ih)+'" x2="'+(W-m.r)+'" y2="'+(m.t+ih)+'"/>');
  g.push('<line class="ieaxis" x1="'+m.l+'" y1="'+m.t+'" x2="'+m.l+'" y2="'+(m.t+ih)+'"/>');
  g.push('<text class="ietitle" x="14" y="15">'+(spec.title||'')+'</text>');
  if(spec.unit)g.push('<text class="ielabel" x="'+(m.l-7)+'" y="'+(m.t-8)+'" text-anchor="end">'+spec.unit+'</text>');
  g.push('<text class="ielabel" x="'+(W-m.r)+'" y="'+(H-6)+'" text-anchor="end">epoch</text>');
  spec.series.forEach(function(s,i){
    if(!s.pts.length)return;
    var d=s.pts.map(function(p,j){return (j?'L':'M')+X(p[0]).toFixed(1)+' '+Yp(Y(p)).toFixed(1);}).join(' ');
    // stroke/fill go through style=, not a presentation attribute: only a CSS
    // declaration may carry var(), so a spec colour can name a theme variable
    g.push('<path class="ieline" style="stroke:'+s.color+'" d="'+d+'"/>');
    if(s.label&&spec.series.length>1)
      g.push('<text class="ielegend" style="fill:'+s.color+'" x="'+(m.l+8+i*92)+'" y="'+(m.t+12)+'">■ '+s.label+'</text>');});
  // 每条曲线的最高点与最低点：▲ = 最高，▼ = 最低，颜色跟随该条曲线。
  // 由渲染器从 series 自己算出来，所以两个页面、所有图表都自动生效。
  // 文字带一圈背景色描边（paint-order:stroke）：两臂对照用的是纯蓝 #0000ff /
  // 纯红 #ff0000，在深色底上直接写字几乎看不清，描边后任何线色都能读。
  // 注意：回调参数不能叫 m —— 会遮住外层的边距对象 m{t,r,b,l}，两个夹回判断
  // 就变成 NaN 比较而永远不生效（这是原来真实存在的 bug）。
  spec.series.forEach(function(s){
    if(s.pts.length<2)return;
    var hi=s.pts[0],lo=s.pts[0];
    s.pts.forEach(function(p){if(p[1]>hi[1])hi=p;if(p[1]<lo[1])lo=p;});
    [[hi,'\u25B2',-7],[lo,'\u25BC',13]].forEach(function(mk){
      var p=mk[0],sym=mk[1],dy=mk[2],px=X(p[0]),py=Yp(Y(p));
      if(dy<0&&py+dy<m.t+18)dy=14;       // 会压住图例行（或标题）就翻到点下方
      if(dy>0&&py+dy>m.t+ih)dy=-9;       // 贴底时翻到点上方，别压住 x 轴刻度
      var tx=px,anc='middle';            // 左右两端把标签夹回绘图区内，别越出坐标轴
      if(px<m.l+34){tx=m.l+2;anc='start';}
      else if(px>W-m.r-34){tx=W-m.r-2;anc='end';}
      g.push('<circle cx="'+px.toFixed(1)+'" cy="'+py.toFixed(1)+'" r="3.2" style="fill:'+s.color+
             ';stroke:var(--ie-bg,#0c0f13);stroke-width:1.2"/>');
      g.push('<text x="'+tx.toFixed(1)+'" y="'+(py+dy).toFixed(1)+'" text-anchor="'+anc+'" style="fill:'+
             s.color+';stroke:var(--ie-bg,#0c0f13);stroke-width:2.6;paint-order:stroke" font-size="10">'+
             sym+ieNum(p[1])+'@'+Math.round(p[0])+'</text>');
    });
  });
  g.push('<g class="iegg"><line class="ieguide" style="display:none" y1="'+m.t+'" y2="'+(m.t+ih)+'"/></g>');
  g.push('<g class="iedots"></g>');
  g.push('<rect class="iehit" x="'+m.l+'" y="'+m.t+'" width="'+iw+'" height="'+ih+'" fill="transparent" style="cursor:crosshair"/>');
  var svg='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'">'+g.join('')+'</svg>'+
          '<div class="ietip"></div>';
  el.innerHTML=svg;
  var svgEl=el.querySelector('svg'),tip=el.querySelector('.ietip'),
      guide=el.querySelector('.iegg line'),dots=el.querySelector('.iedots'),
      hit=el.querySelector('.iehit');
  // go through the browser's own transform: hand-rolled px/unit ratios drift as soon
  // as the card reflows, which is what made the marker land far from the cursor
  function toUser(ev){
    if(!svgEl.getScreenCTM) return {x:ev.clientX,y:ev.clientY};
    var pt=svgEl.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;
    var q=pt.matrixTransform(svgEl.getScreenCTM().inverse());return {x:q.x,y:q.y};}
  function toPage(ux,uy){
    var pt=svgEl.createSVGPoint();pt.x=ux;pt.y=uy;
    return pt.matrixTransform(svgEl.getScreenCTM());}
  function nearest(u){var best=null,bd=1e18;      // 2-D: the point you are actually pointing at
    all.forEach(function(p){var dx=X(p[0])-u.x,dy=Yp(Y(p))-u.y,d=dx*dx*0.55+dy*dy*1.45;
      if(d<bd){bd=d;best=p;}});return best;}
  hit.addEventListener('mousemove',function(ev){
    var box=el.getBoundingClientRect(),u=toUser(ev),p=nearest(u);
    if(!p)return;
    var x=X(p[0]),y=Yp(Y(p));
    guide.setAttribute('x1',x);guide.setAttribute('x2',x);guide.style.display='';
    var html='<b>epoch '+Math.round(p[0])+'</b>';
    spec.series.forEach(function(s){
      var q=s.pts.filter(function(v){return v[0]===p[0];})[0];if(!q)return;
      html+='<br><span style="color:'+s.color+'">■</span> '+
            (spec.series.length>1?s.label+' ':'')+'<b>'+ieNum(q[1])+'</b>';});
    if(p[2]){if(p[2].lr!=null)html+='<br>lr '+p[2].lr.toExponential(2);
             if(p[2].ts)html+='<br>'+p[2].ts;}
    tip.innerHTML=html;tip.style.opacity=1;
    dots.innerHTML=spec.series.map(function(s){
      var q=s.pts.filter(function(v){return v[0]===p[0];})[0];if(!q)return '';
      return '<circle cx="'+X(q[0]).toFixed(1)+'" cy="'+Yp(Y(q)).toFixed(1)+
             '" r="3.6" style="fill:'+s.color+'" stroke="var(--ie-bg,#0c0f13)" stroke-width="1.4"/>';}).join('');
    var anchor=toPage(x,y);                        // anchor on the point, not on the cursor
    var lx=anchor.x-box.left+11,ly=anchor.y-box.top-tip.offsetHeight-8;
    if(lx+tip.offsetWidth>box.width-6)lx=anchor.x-box.left-tip.offsetWidth-11;
    if(ly<0)ly=anchor.y-box.top+12;
    tip.style.left=Math.max(2,lx)+'px';tip.style.top=Math.max(2,ly)+'px';});
  hit.addEventListener('mouseleave',function(){
    tip.style.opacity=0;guide.style.display='none';dots.innerHTML='';});
}
function ieRenderCharts(root){
  (root||document).querySelectorAll('.iechart:not([data-ied])').forEach(function(el){
    el.setAttribute('data-ied','1');ieDraw(el);});}
window.addEventListener('resize',function(){
  document.querySelectorAll('.iechart').forEach(function(el){
    el.removeAttribute('data-ied');});ieRenderCharts();});
"""


def chart_div(spec: dict) -> str:
    """HTML for one chart; the page must include CHART_JS and call ieRenderCharts()."""
    return f'<div class="iechart" data-c="{quote(json.dumps(spec, separators=(",", ":")), safe="")}" ' \
           f'style="min-height:{spec.get("h", 210) - 40}px"></div>'


def phase_series(rows: list[dict], key: str, palette=PALETTE,
                 colors: dict | None = None) -> list[dict]:
    """One series per phase, so a warm-up block and a re-run never share an x axis.

    ``colors`` maps a phase name to an explicit colour, overriding the palette; that
    is how a caller pins one curve to one colour (e.g. blue = no IWO, red = IWO).
    """
    groups: dict[str, list] = {}
    for r in rows:
        value = r.get(key)
        if value is None:
            continue
        groups.setdefault(r.get("phase", "train"), []).append(
            [r.get("epoch", 0), value, {"lr": r.get("lr"), "ts": r.get("ts")}])
    return [{"label": name, "color": (colors or {}).get(name) or palette[i % len(palette)],
             "pts": pts}
            for i, (name, pts) in enumerate(groups.items())]


def chart_for_rows(rows: list[dict], key: str, *, title: str, unit: str = "",
                   log: bool = False, height: int = 210, palette=PALETTE,
                   colors: dict | None = None,
                   xstep: int = EPOCH_TICK_STEP) -> str:
    series = phase_series(rows, key, palette, colors)
    # 最高/最低点由 CHART_JS 从 series 自动标注，spec 里不再带 best
    return chart_div({"title": title, "unit": unit, "log": log, "h": height,
                      "xstep": xstep, "series": series})
