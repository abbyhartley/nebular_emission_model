import sys
p = sys.argv[1]
lines = open(p).read().splitlines()
# per-panel robust limits (IQR fence: keep all real data, drop only far outliers),
# injected at the top of hex_panel so each panel hugs its own distribution.
block = [
    "    import numpy as _np_pp",
    "    _xy = _np_pp.concatenate([_np_pp.asarray(x, float).ravel(), _np_pp.asarray(y, float).ravel()])",
    "    _xy = _xy[_np_pp.isfinite(_xy)]",
    "    _q1, _q3 = _np_pp.percentile(_xy, [25, 75]); _fe = 3.0 * (_q3 - _q1)",
    "    _kp = _xy[(_xy >= _q1 - _fe) & (_xy <= _q3 + _fe)]",
    "    if _kp.size == 0: _kp = _xy",
    "    _loo = float(_kp.min()); _hii = float(_kp.max()); _pd = 0.04 * (_hii - _loo)",
    "    lo = _loo - _pd; hi = _hii + _pd",
]
out, done = [], False
for ln in lines:
    out.append(ln)
    if (not done) and ln.lstrip().startswith("def hex_panel"):
        out.extend(block)
        done = True
txt = "\n".join(out)
txt = txt.replace('sharex="col"', "sharex=False").replace('sharey="col"', "sharey=False")
txt = txt.replace("labelleft=False", "labelleft=True")
open(p, "w").write(txt + "\n")
print(f"perpanel patched {p}: hex_panel_found={done}")
