import sys
p = sys.argv[1]
lines = open(p).read().splitlines()
ins = ('from functools import partial as _pf\n'
       'import jax.numpy as _jnp_rob\n'
       'from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces\n'
       '_ROBUST_INV = _r2i(_pf(_bces, midpoint=_jnp_rob.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))')
out, done = [], False
for line in lines:
    out.append(line)
    if (not done) and ("block_neural_autoregressive_flow" in line) and (line.strip().startswith("from") or " import " in line):
        out.append(ins)
        done = True
open(p, "w").write("\n".join(out) + "\n")
print(f"patched {p}: inverter def inserted = {done}")
