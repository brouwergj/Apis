# HANDOVER: Crazyflie Controller Port into NUC

## 1. Big picture

We are working inside a larger project called **Apis**, which includes a subsystem called **NUC (Networked Unity Controller)**.

**Core NUC design principle:**
- **Python is the authoritative control & simulation process**
- **Unity is visualization only**
- Python owns time, physics, and control
- Unity only consumes telemetry and renders state

This separation is **intentional and non-negotiable**.

---

## 2. What already exists (do NOT rebuild)

### NUC architecture (already implemented and working)

#### Python side
- Deterministic, fixed-`dt` simulation loop
- Toy quadrotor dynamics (gravity, drag, thrust lag, simple attitude response)
- Engineered cascaded controller:
  - position → attitude → thrust + yaw-rate
- CSV logging per timestep
- Optional real-time pacing
- Optional UDP telemetry (latest-only, drop-friendly)
- WebSocket server:
  - Accepts exactly one `START` command
  - Sends `ACK`
  - Runs one episode
  - Sends `DONE`
  - Exits

This **one-shot process lifecycle** is deliberate.

#### Unity side
- Starts/stops Python as an OS process
- Sends RunConfig via WebSocket
- Receives pose telemetry via UDP
- Applies pose to a Crazyflie GameObject
- No physics
- No timing authority

All of this is stable and already working.

---

## 3. What we are building now (goal)

We want to add a **Crazyflie-firmware-faithful controller implementation** to NUC, **alongside** the existing toy controller.

Important clarifications:

- We are **NOT** building a “Crazyflie-style” controller.
- We are **re-implementing the actual Crazyflie controller architecture**, as found in the Bitcraze `crazyflie-firmware` repository.
- This new controller must:
  - use the same RunConfig interface
  - run inside the same Python simulation loop
  - produce the same logging
  - allow apples-to-apples comparisons:
    - toy controller vs Crazyflie PID
    - later: Mellinger / INDI / learned surrogate

Unity must remain unchanged.

---

## 4. Scope boundaries (very important)

### What we DO want to port
- Controller architecture
- Controller update order
- Controller math
- PID structure
- Position + attitude cascade
- Rate supervision and filtering
- Control abstractions:
  - roll / pitch
  - yaw-rate
  - collective thrust

### What we EXPLICITLY do NOT port
- Motor mixing
- PWM / ESC logic
- RTOS (FreeRTOS)
- Radio / CRTP
- Sensors / decks
- Logging infrastructure
- Parameter server
- Power distribution

These are intentionally *out of scope*.

---

## 5. Crazyflie controller architecture (as discovered)

The Crazyflie firmware has a **clean, layered control spine**:

stabilizer.c
→ controller.c (controller selection & dispatch)
→ controller_pid.c (or mellinger / indi / lee / brescianini)
→ position_controller_pid.c
→ attitude_pid_controller.c
→ pid.c
→ filter.c
→ rateSupervisor.h


### Controllers present in firmware
- PID (baseline, first target)
- Mellinger
- INDI
- Lee
- Brescianini

All controllers implement the **same interface** and plug into the same stabilizer loop.

---

## 6. Controller data contracts (critical)

Defined in `stabilizer_types.h` and **must be mirrored conceptually in Python**:

- `state_t`
  - position
  - velocity
  - attitude (quaternion / Euler)
  - angular rates
- `setpoint_t`
  - desired position / velocity / attitude / yaw-rate / thrust
  - mode flags
- `control_t`
  - roll
  - pitch
  - yaw-rate
  - thrust

These structs define the **contract** between estimator → controller → actuators.

---

## 7. Files identified as relevant

### Core loop & dispatch
- `stabilizer.c`
- `controller.c`

### Controllers
- `controller_pid.c`
- `controller_mellinger.c`
- `controller_indi.c`
- `controller_lee.c`
- `controller_brescianini.c`

### Cascade layers
- `position_controller_pid.c`
- `attitude_pid_controller.c`

### Primitives & helpers
- `pid.c` / `pid.h`
- `filter.c` / `filter.h`
- `num.h`
- `rateSupervisor.h`

### Constants & defaults
- `physicalConstants.h`
- `platform_defaults.h`

### Interfaces
- `controller.h`
- `controller_pid.h`
- `position_controller.h`
- `attitude_controller.h`
- `stabilizer_types.h`

### Notes
- `autoconf.h` is build-generated in firmware and does **not** exist in the repo.
- RTOS, motor, sensor, and logging includes can be stubbed or ignored.

---

## 8. Python implementation intent

### Structural fidelity over literal translation

The Python controller should:
- mirror the **update order**
- mirror the **responsibility boundaries**
- mirror the **math and semantics**

But be:
- idiomatic Python
- readable
- simulation- and learning-friendly

### Suggested Python layout

crazyflie_controller/
├── types.py
├── pid.py
├── filter.py
├── rate_supervisor.py
├── position_controller.py
├── attitude_controller.py
├── controller_pid.py
├── controller_dispatch.py
└── crazyflie_controller.py


This must plug into the existing NUC loop **without changing lifecycle or orchestration**.

---

## 9. Recommended implementation order

1. Reconstruct call graph from:
   - `stabilizer.c`
   - `controller.c`
2. Implement PID primitive + filters
3. Implement position controller
4. Implement attitude controller
5. Implement Crazyflie PID controller
6. Verify sign conventions, saturation, and ranges
7. Extend to Mellinger / INDI only after PID is correct

---

## 10. Success criteria

This effort is successful when:

- The Python Crazyflie PID controller:
  - runs inside the NUC loop
  - produces stable flight in toy dynamics
  - uses the same abstractions as firmware
- Switching controllers requires **no Unity changes**
- Structure is clean enough to support:
  - learned controllers
  - controlled comparisons

---

## 11. Guiding philosophy

This is **not** about perfect physical fidelity.

It *is* about:
- architectural fidelity
- controller semantics
- clean separation of concerns
- reproducibility and comparability

If a trade-off is required:
> **clarity and structure beat micro-optimizations**

---

## 12. Final note

All discovery work is complete.

This is now a **straight engineering task**:
- port
- structure
- verify

Prioritize:
- correctness of control flow
- matching firmware intent
- readable, maintainable Python