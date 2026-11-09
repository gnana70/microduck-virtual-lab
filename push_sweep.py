import vlab, mujoco, numpy as np, json, math, sys
res = {}
starts = [1.0 + 0.07*i for i in range(10)]
for kind in ['hd1910','xl330']:
  for f in [6, 8, 10, 12, 15]:
    falls = 0
    for s in starts:
        m,d,c,p = vlab.build(kind, render=(320,240))
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
        fell = False
        while d.time < s + 3.0:
            d.xfrc_applied[bid,:3] = [0, f, 0] if s <= d.time < s+0.1 else 0
            vlab.step_policy(m,d,c,p,0,0,ticks=1)
            pos,tilt,_ = vlab.trunk_state(m,d)
            if tilt > 45 or pos[2] < 0.06: fell = True; break
        falls += fell
    res[f'{kind}_{f}'] = falls
    print(kind, f, 'N side: fell', falls, '/', len(starts), flush=True)
json.dump(res, open('results/push_sweep.json','w'))
