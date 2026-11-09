"""Battery-voltage sweep for the HD-1910 model: same walk command (vx = 0.25) with a simple heading hold
(like small corrections on the steering stick). Writes results/battery_steered.json."""
import json, math
import numpy as np
import vlab

out = {}
for vin in [6.8, 7.0, 7.2, 7.4, 7.7, 8.0, 8.4]:
    m, d, c, p = vlab.build("hd1910", render=(320, 240), vin=vin)
    if vin < 7.0:
        c.vin_min = 6.5          # let the effective voltage follow the battery below the usual 7.0 V floor
    vlab.step_policy(m, d, c, p, ticks=50)
    pos0, _, y0 = vlab.trunk_state(m, d)
    for _ in range(500):          # 10 s at 50 Hz
        _, _, yaw = vlab.trunk_state(m, d)
        err = (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
        vlab.step_policy(m, d, c, p, 0.25, float(np.clip(-1.0 * err, -0.6, 0.6)), ticks=1)
    pos, tilt, _ = vlab.trunk_state(m, d)
    out[str(vin)] = round(float(np.hypot(*(pos[:2] - pos0[:2]))), 3)
    print(vin, "V:", out[str(vin)], "m in 10 s")
json.dump(out, open("results/battery_steered.json", "w"))
