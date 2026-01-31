# Crazyflie Autonomous Control Experiments

## Overview

This project explores **autonomous drone behaviors trained with reinforcement learning (RL)**, using a **Crazyflie 2.1** quadcopter as the primary real-world platform.

A core design principle is that **RL does not learn how to fly the drone**.  
Low-level stabilization, state estimation, and motor control are delegated to an existing **flight controller stack**, while learning-based components operate strictly at a **higher command level** (e.g. velocity, position, or trajectory intent).

The project is structured to support:

- real hardware experiments,
- software-in-the-loop (SITL) testing with identical APIs,
- physics-based simulation (Isaac Sim / Unity),
- and optional side experiments in controller modeling for learning purposes.

---

## Core Control Architecture

### Flight controller vs autonomy (intentional split)

The Crazyflie firmware already provides:

- onboard **state estimation** (IMU + additional sensors if present),
- a **cascaded control stack** (rate → attitude → velocity → position),
- a **commander interface** that accepts high-level setpoints.

This project **reuses that controller stack as-is**.

The autonomy system (manual input or RL policy):

- never controls motors directly,
- never runs at flight-controller rates,
- issues only **high-level commands / setpoints**.

This mirrors how manual flight via the Crazyflie client already works.

---

## Timing Model and Command Streaming

### Human input vs command rate (important clarification)

When flying manually:

- human/gamepad inputs update slowly and irregularly,
- the client application continuously **streams setpoints** to the drone at a fixed rate,
- stability comes from **continuous command streaming**, not from human input frequency.

### Implication for autonomy / RL

Autonomous control must follow the same pattern.

The system is explicitly split into **two loops**:

#### 1. Policy loop (slow)

- Runs at 10–30 Hz (vision-based RL) or up to ~50 Hz (state-based).
- Produces *intent* (desired command).
- May be learned (RL, imitation, etc.).

#### 2. Command streaming loop (fast)

- Runs at 50–200 Hz (exact value TBD).
- Continuously sends setpoints to the drone.
- Holds, interpolates, or rate-limits between policy updates.
- Is always active while the drone is armed.

If command streaming stops, failsafe behavior is expected.

This loop separation is **non-optional** and applies to:

- real hardware,
- CrazySim (SITL),
- Unity / Isaac Sim backends.

---

## Simulation Strategy

### Software-in-the-Loop (SITL)

**CrazySim (SITL)** runs the *unmodified Crazyflie flight-controller firmware* on a host machine.

Benefits:

- identical commander API to real hardware,
- validates timing, saturation, safety logic, and control semantics,
- ideal for testing autonomy logic without crashes.

Limitations:

- simplified physics and sensing,
- not suitable for perception-heavy learning.

### Physics Simulation

Physics-based simulators (Isaac Sim / Unity) are used to:

- introduce richer sensors (RGB, LiDAR, etc.),
- train perception-heavy policies,
- apply domain randomization.

The autonomy API is kept identical across all backends.

---

## Action Space (Deferred Decision)

The exact command representation used by the policy is **intentionally deferred**.

Candidates include:

- velocity + yaw-rate setpoints,
- position setpoints (requires strong localization),
- high-level trajectory primitives.

The decision will be made after:

- basic autonomy tests in CrazySim,
- stability and latency evaluation,
- sensor configuration is finalized.

---

## Safety and Non-Goals

### Explicit non-goals

- RL will **not** learn motor control or thrust mixing.
- RL will **not** bypass the Crazyflie controller stack.
- RL will **not** be responsible for failsafe recovery.
- Controller learning is **not required** for this platform.

### Safety measures

- command saturation and rate limiting,
- hard altitude and velocity limits,
- manual kill / disarm always available.

---

## Optional Exercise: Neural Surrogate Controller (Learning the Controller)

As a **separate, optional exercise**, this project includes experimenting with **learning a neural surrogate of a known controller**.

### Motivation

In some commercial or certified drones, the flight controller is closed-source and inaccessible.  
In such cases, one may be forced to **learn a surrogate controller from data**.

While this is *not needed* for Crazyflie (which is open and accessible), it is included here as:

- a learning exercise,
- a way to understand differentiable control,
- a comparison point to black-box system identification approaches.

### Exercise outline

1. Define a **simple but realistic 3D quadcopter model**:
   - position (x, y, z), velocity,
   - roll, pitch, yaw,
   - thrust-based motion with gravity and drag.

2. Implement a **cascaded controller**:
   - outer loop: position → desired acceleration,
   - inner loop: desired acceleration → roll, pitch, thrust, yaw rate.

3. Treat this controller as a **black box** and generate flight data.

4. Train a neural network (e.g. using JAX) to:
   - approximate the controller’s input–output behavior,
   - using supervised learning (behavioral cloning).

5. Validate the surrogate **in closed loop**.

### Important note

This exercise is **conceptually separate** from the main project:

- It exists to understand controller learning under constraints.
- It is *not* intended to replace or improve the Crazyflie controller.
- Results from this exercise will not be deployed to real hardware.

---

## Roadmap (High-Level)

1. Manual flight + hardware validation  
2. CrazySim SITL integration  
3. Autonomy API definition  
4. Basic autonomy tests (state-based)  
5. Physics simulation + richer sensing  
6. (Optional) Neural surrogate controller exercise  
7. Sim-to-real experiments (sensor-dependent)

---

## Philosophy

> **RL decides what to do.  
> The flight controller decides how to stay airborne.  
> Timing, safety, and control hierarchy are never learned by accident.**
