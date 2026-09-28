# Quarter-Car RL Active Suspension — Full Build Spec (Course + SIH)

## Context
Two projects share one technical core:
- **Course mini-project** (MBD, mentor: Mrs. Mrunalini Bhandarkar) — quarter-car RL
  simulation, standard 4-wheel-style parameters, idealized active-force case is fine
  since this is a simulation-only academic deliverable.
- **SIH 2026 submission** (team "Synapse") — two-wheeler pivot, front-fork vertical
  dynamics only (no lean/cornering), with a physical semi-active hardware demo
  (ESP32 + sensors + actuator + live dashboard). SAC is the locked algorithm for both.

Same physics structure, different parameters and different action-space realism.
Build one shared core and two thin project layers on top of it, in one directory,
so nothing gets duplicated or drifts out of sync between the two deliverables.

## Repo layout
```
project_root/
  core/
    dynamics.py            # equations of motion, solve_ivp wrapper, parameterized
    road_profiles.py         # bump, ISO 8608 random generator
    env.py                     # Gymnasium env; semi-active + active action-space variants
    baseline_controllers.py      # passive, PID/LQR/skyhook
    train_sac.py                   # SAC training loop, checkpointing, reward logging
    evaluate.py                      # RMS metrics, robustness sweep
    visualize.py                       # all plotting functions

  course_project/
    notebook.ipynb           # thin orchestration, no markdown cells, short plain comments only
    config.py                  # standard quarter-car parameter set, active-force case allowed

  sih_project/
    notebook.ipynb           # re-parameterized training/eval, semi-active primary case
    config.py                  # front-fork parameter set (measured or literature-sourced)
    firmware/                  # ESP32 embedded code
      inference.cpp / .h         # hand-written forward pass for exported policy weights
      sensors.cpp / .h             # MPU6050 + linear position sensor reads
      actuator_driver.cpp / .h       # motor driver PWM control, bounded semi-active range
      main.ino                         # sensor-read -> inference -> actuator-drive loop
    dashboard/                 # live telemetry web view
      server side (Python/Node)  # WiFi telemetry ingestion from ESP32
      index.html                   # live accel/position/comfort-metric view, mode switch
    docs/
      bom.md                     # bill of materials
      system_architecture.md       # architecture writeup
      scope_boundaries.md            # honest limitations, stated proactively not defensively
```

Everything under `core/` is written once and imported by both `course_project/` and
`sih_project/` — parameter sets and action-space choice differ, equations don't.

---

## SHARED CORE SPEC

### Physical model
2-DOF quarter-car / quarter-motorcycle (same structure, different constants):
```
ms*x1'' = -ks*(x1-x2) - c*(x1'-x2') + Fa
mu*x2'' =  ks*(x1-x2) + c*(x1'-x2') - kt*(x2-zr) - Fa
```
- Course project: standard car-style quarter-car parameters (illustrative/academic values
  are fine here, e.g. `ms≈250kg, mu≈50kg, ks≈16000N/m, c≈1000Ns/m, kt≈190000N/m`)
- SIH project: front-fork parameters, sourced by direct measurement where possible
  (static spring-rate test: compress fork known distance under known load, `k=F/x`;
  weigh front wheel + lower fork assembly for `mu`; front-end static load share for
  `ms`), falling back to literature values (Vico et al. 2021; Segla & Roy 2020) only
  if a real bike/scooter isn't accessible. Rear mono-shock is deliberately out of
  scope — its lever-ratio linkage doesn't map cleanly onto this model, front fork does.

Road inputs: deterministic bump (clean demo plots) + ISO 8608 class B random profile
(robustness evaluation). Integration via `scipy.integrate.solve_ivp`, fixed step.

### Action space (two variants, both in `env.py`)
- **Semi-active** (primary claim, matches actual hardware): 1D continuous, bounded
  `[c_min, c_max]`, `c_min > 0` — dissipative only, physically what a variable damper
  can do. This is the SIH hardware-aligned case.
- **Active** (idealized, comparison/benchmark case only): 1D continuous, bounded
  `[-Fa_max, Fa_max]`. Fine for the course project's simulation-only deliverable;
  in the SIH context this is explicitly labeled as a non-hardware upper-bound
  comparison, never presented as what the rig does.

### Reward
`r = -(w1*x1_accel^2 + w2*(x1-x2)^2 + w3*(x2-zr)^2 + w4*control_effort^2)`

### Baseline controllers
Passive (fixed `c`, `Fa=0`) and a classical baseline (PID or skyhook; SIH strategy
docs also mention LQR as an option) — needed so the RL result has a real comparison.

### Evaluation metrics
- RMS body acceleration (ISO 2631-weighted where time allows — comfort proxy)
- RMS tire deflection (road-holding / safety proxy — ties to MoRTH/NCRB framing)
- RMS suspension travel (rattle-space proxy)
- Robustness: trained policy evaluated on road inputs/profiles it wasn't trained on

### Visualization
Time-domain overlays (passive vs controlled), RMS-metric comparison bars, SAC
training reward curve. Labeled axes/units throughout — these are the plots that get
explained to a mentor or a judge.

### Code delivery constraint (course_project notebook specifically)
No markdown cells, no headers-as-cells — short plain code comments only, written
like a student wrote them. All plots inline. This does not apply to the SIH
firmware/dashboard code, which should be normally commented for a real embedded/web
codebase.

---

## SIH-SPECIFIC ADDITIONS

### System architecture
Training happens offline in Python (SAC via `core/`, `sih_project/config.py`
parameters). The trained actor network is exported and its forward pass
hand-written in C/C++ for the ESP32 (small network — 2 hidden layers, 64–256
units — no need for TensorFlow Lite Micro or similar unless this turns out to be
a bottleneck, which is unlikely at this size). The ESP32 reads the accelerometer
and position sensor continuously, runs inference, drives the actuator through the
motor driver, and streams telemetry over WiFi to the dashboard, which also hosts
the passive/baseline/RL mode switch for the live comparison demo.

### Physical rig concept (for context — physical build is Amey's track, not code)
Vertical column: base plate, a guided mass (sprung mass analog) on a spring, a
small actuator/damper element between mass and base (the controllable element),
on a platform that can simulate a bump/road input. Doesn't need to resemble an
actual fork — it needs to demonstrate the control loop convincingly.

### Bill of materials (approximate, verify current pricing before buying)
| Component | Purpose | Approx. cost (INR) |
|---|---|---|
| ESP32 dev board (WROOM-32) | Sensor read + inference + WiFi | 350–600 |
| MPU6050 accelerometer/gyro | Body acceleration | 120–250 |
| Linear potentiometer / magnetic linear position sensor | Suspension deflection | 150–500 |
| Small 12V linear actuator (position feedback) or geared DC motor + lead screw | Controllable damping element | 800–2500 |
| Motor driver (e.g. BTS7960 H-bridge) | Drives actuator from ESP32 PWM | 150–400 |
| Load cell + HX711 (optional) | Tire-load proxy | 300–600 |
| 12V power supply | Powers actuator + electronics | 500–1500 |
| Frame materials (aluminum extrusion / 3D-printed / MS tube) | Rig structure | 1000–3000 |
| Misc (wiring, connectors, perfboard, enclosure) | Assembly | 500–1000 |

Rough total ₹4,000–10,500. Verify against current listings before finalizing.

### Firmware spec (`sih_project/firmware/`)
- `sensors.cpp/.h`: read MPU6050 (I2C) and linear position sensor (analog/I2C
  depending on part chosen)
- `inference.cpp/.h`: hand-written forward pass — a few matrix-vector multiplies
  plus activation functions — taking exported weight arrays from the trained
  SAC actor, bounded output mapped to `[c_min, c_max]` (semi-active only, this
  board never outputs an arbitrary bidirectional force)
- `actuator_driver.cpp/.h`: translate the bounded damping command into PWM for
  the motor driver
- `main.ino`: ties it together — read sensors, run inference, drive actuator,
  push telemetry over WiFi, all in the real-time loop

### Dashboard spec (`sih_project/dashboard/`)
- Ingests WiFi telemetry from the ESP32 (accelerometer reading, position,
  computed comfort metric, current control mode)
- Live web view of the above, plus a mode switch (passive / baseline / RL) so a
  judge can watch the comparison happen in real time
- Format of the telemetry payload should be agreed between firmware and
  dashboard code up front (simple JSON over WiFi is sufficient)

### Honest scope boundaries — state these proactively when presenting, not just if asked
- Small-scale proof of concept, not a vehicle-ready system — forces, masses,
  travel are all scaled down from a real motorcycle
- The physical rig demonstrates **semi-active** control only; any active-force
  simulation results are a separate idealized comparison, not what the hardware does
- Sim-to-real gap: a policy trained purely in simulation may not behave
  identically on the rig (real friction, sensor noise, actuator nonlinearity
  aren't in the simulator) — budget a short on-rig tuning/safety-bounding pass

### Task ownership (context, not all of this is code Claude Code produces)
- **RL/Simulation (Yajat, Nikhilesh)**: everything in `core/` and both
  `config.py`/`notebook.ipynb` pairs — this is the part Claude Code builds
- **Mechanical/Electronics (Amey)**: physical rig build, wiring, sensor
  mounting — physical work, not code; `docs/bom.md` and `docs/
  system_architecture.md` support this but don't replace it
- **Embedded/Firmware (Yash)**: owns `sih_project/firmware/` — Claude Code can
  scaffold this, but final on-board tuning is a physical/iterative task
- **Verification/Dashboard (Shardul)**: Scilab cross-verification (outside this
  repo's scope, separate tool) + `sih_project/dashboard/` — Claude Code can
  build the dashboard code
- **Presentation/Impact (Harshita)**: pitch narrative, not code

---

## 50% CHECKPOINT — due tomorrow, `course_project/` only

1. `core/dynamics.py` + `core/road_profiles.py` — working ODE model, bump input,
   sane plots (no NaNs/instability)
2. Passive baseline simulated and plotted (bump + random road)
3. `core/env.py` — semi-active variant only for now, `reset()`/`step()` working
4. Sanity check: random policy episode + skyhook baseline, plotted, stable
5. SAC training started (`core/train_sac.py`), modest timestep budget, reward
   curve plotted even if not converged
6. One comparison plot: passive vs current (partially trained) SAC policy

Not expected tomorrow: convergence, active-force variant, robustness sweep,
ISO 2631 weighting, anything under `sih_project/` — that track isn't due yet.

---

## FULL SCOPE — remaining work after checkpoint

**Course project (`core/` + `course_project/`)**
- Finish semi-active SAC training to convergence, hyperparameter tuning
- Add active-force variant as a labeled idealized comparison case
- Full robustness evaluation across road classes/speeds
- ISO 2631-weighted comfort metric, reward-weight sensitivity/ablation
- Final report consolidation, same defensible-novelty framing already agreed
  (a scoped contribution, not "unexplored territory")

**SIH project (`sih_project/`)**
- Re-run training/eval with real front-fork parameters once measured
- Build out `firmware/` — sensor reads, hand-written inference, actuator driver,
  main loop (placeholder/random-weight policy first, to decouple from RL track)
- Build out `dashboard/` — telemetry ingestion, live view, mode switch
- Physical rig assembly, wiring, on-rig tuning pass (Amey/Yash, physical work)
- Handoff point: exported trained policy weights from `core/train_sac.py` output
  feed directly into `firmware/inference.cpp`
- Scilab cross-verification (Shardul, separate toolchain, not in this repo)
