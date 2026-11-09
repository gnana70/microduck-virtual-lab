# Microduck virtual lab

The open-source Microduck (Pollen Robotics / Hugging Face) and the community **microduck-replica** build, tested in simulation before buying parts: the real MJCF + STL model in MuJoCo, servos simulated with Rhoban's BAM M6 models, and Pollen's official `velstand` walking policy.

Companion code for the Singularity Feed video *Build Microduck at home*.

## What it tests

| Script | Test | Result file |
|---|---|---|
| `physics_tests.py` | model census (parts, servos, bearings, mass, height), hold test without a policy, walking, turning, pushes | `results/physics.json` |
| `push_sweep.py` | sideways push for 0.1 s at 10 timings per force, both servo models | `results/push_sweep.json` |
| `battery_steered.py` | walking distance in 10 s vs battery voltage, HD-1910 model, with heading hold | `results/battery_steered.json` |

Two servo models, both BAM M6:

* **Dynamixel XL330** (Pollen's servo): bundled `xl330/m6` parameters, firmware P = 200, 7.4 V, as in `microduck_rl/scripts/infer_policy.py`.
* **Feetech HD-1910** (the replica's servo): LuwuDynamics `1910_m6.json` (as shipped in microduck-replica), firmware P = 5 (also what the real replica writes), 7.7 V nominal, 7.0 V floor, as in microduck-replica's `microduck_hd1910_env_cfg.py`. The Feetech goal limiter is seeded from the measured pose on reset, like the replica's `FeetechBamActuator` fix.

## Results (9 Nov 2026, MuJoCo 4.0.0)

* 70 visual parts (38 types), 15 servos, 14 bearings, 737 g, 25.5 cm standing.
* Hold test, no policy: XL330 model at P = 200 sagged and toppled (tilt 81°); HD-1910 model at P = 5 stood (2°). XL330 at P = 400 also stands, so this is about holding stiffness.
* Walk straight (vx = 0.25 m/s command, 10 s) with Pollen's velstand policy: XL330 0.61 m forward, 21° heading drift; HD-1910 0.27 m forward, 0.36 m sideways, 112° heading change (never fell). With a simple heading hold the HD-1910 model walks 0.65 m in 15 s, straight.
* Push (HD-1910): 8 N for 0.1 s, 0/10 falls; 10 N, 9/10 falls, all 9 recovered within 5 s; 12 N, 10/10 falls.
* Knee torque while walking (HD-1910): peak 0.43 N m (29 % of the 1.47 N m stall spec), RMS 0.16 N m (43 % of the 0.36 N m rated load).
* Battery (steered): 6.8 V → 0.01 m in 10 s; 7.0–8.4 V → 0.42–0.48 m.

Simulation only: no claim about a physical duck. The replica project's own hardware validation (standing, walking, forward roll and recovery) is in its repository.

## Run it

```bash
pip install mujoco==4.0.0 onnxruntime numpy colorama
mkdir -p third_party policies
git clone https://github.com/pollen-robotics/microduck_rl third_party/microduck_rl        # tested at bae39f5
git clone https://github.com/Rhoban/bam third_party/bam && git -C third_party/bam checkout 62bd8ce12154340be97e06f7f41a0ca8f116d967
# Pollen's walking policy: download velstand.onnx from https://huggingface.co/pollen-robotics/microduck-policies into policies/
#   (we used sha256 2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492)
python physics_tests.py && python push_sweep.py && python battery_steered.py
```

Set `MUJOCO_GL=egl` on a headless Linux machine.

## Credits and licences

* Model and meshes: Pollen Robotics, [microduck_rl](https://github.com/pollen-robotics/microduck_rl) (code Apache-2.0; 3D model CC BY-NC-SA per its README). Not included here; cloned at run time.
* Walking policy: Pollen Robotics `velstand.onnx`. Not included here.
* Servo models: [Rhoban/BAM](https://github.com/Rhoban/bam) (Apache-2.0).
* HD-1910 parameters `params/1910_m6.json`: [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) @ 326d77a (Apache-2.0, licence copy in `params/`), as redistributed by [microduck-replica](https://github.com/fanhao375/microduck-replica).
* This lab's scripts: Apache-2.0.

Not affiliated with Pollen Robotics, Hugging Face or the microduck-replica project.
