# Apis
### Autonomous Systems, Control, and Learning Framework

## Overview

**Apis** is a structured, stepwise framework for developing, understanding, and experimenting with autonomous aerial systems.

The project is intentionally organized as a **sequence of concrete steps**, each building on the previous one.  
Every step produces artifacts (code, data, interfaces) that are required by the next.

The primary real-world reference platform is the **Crazyflie 2.1** quadcopter, but Apis is designed to generalize beyond a single vehicle or simulator.

A guiding principle throughout Apis is:

> **Learning systems do not learn how to keep vehicles airborne.**  
> Stability, state estimation, and safety‑critical control are engineered.  
> Learning happens *on top*, never underneath.

---

## Project Structure

```
Apis
 ├─ Step 1: Nuc
 ├─ Step 2: Brood
 ├─ Step 3: Waggle
 ├─ Step 4: Propolis
 ├─ Step 5: Scopa
 └─ Step 6: Proboscis
```

---

## Step 1 — Nuc  
### Deterministic Control & Simulation Core

Nuc is the nucleus of Apis.

Goal:
- Build a **deterministic simulation environment** running a known flight controller in closed loop with simplified quadcopter dynamics.

Outputs:
- fixed‑step simulation loop
- cascaded “teacher” controller
- reference programs (“how to fly”)
- high‑quality, reproducible datasets

Visualization (Unity) is external and non‑authoritative.

---

## Step 2 — Brood  
### Neural Surrogate Controller Learning

Brood builds directly on the datasets produced by Nuc.

Goal:
- Learn a **neural surrogate** of a flight controller from observed behavior.

Approach:
- treat the controller as a black box
- train models using logged state, reference, and command data
- implement learning pipelines in **JAX**

Brood exists to understand the limits and behavior of learned controllers, not to replace real flight stacks.

---

## Step 3 — Waggle  
### Unity‑Based Controller‑in‑the‑Loop Scaffolding

Waggle introduces structured interaction between deterministic control logic and a real‑time 3D environment.

Goal:
- Use Unity to send high‑level actions
- Receive state estimates over fixed time windows
- Evaluate timing, latency, and action semantics

Unity acts as orchestrator and viewer, never as the source of truth.

---

## Step 4 — Propolis  
### Isaac Sim Integration

Propolis extends the same controller‑in‑the‑loop architecture to **Isaac Sim**.

Goal:
- Operate a virtual Crazyflie controller inside Isaac Sim
- Compare behavior, timing, and sensing with Unity‑based workflows

This step focuses on higher‑fidelity physics and robotics‑oriented simulation.

---

## Step 5 — Scopa  
### Virtual Sensor Expansion & Data Generation

Scopa introduces richer virtual sensing.

Goal:
- Add sensors such as RGB cameras, depth, or LiDAR
- Generate aligned datasets combining controller state, actions, and sensor observations

Unity and Isaac Sim become data generators rather than control authorities.

---

## Step 6 — Proboscis  
### Perception, Fusion, and Active Control Pipelines

Proboscis closes the loop from perception to action.

Goal:
- Build pipelines that connect sensor data to high‑level commands using:
  - SLAM / localization
  - sensor fusion
  - learned or classical policies
  - active perception strategies

This is where full autonomy emerges as a system.

---

## Safety and Non‑Goals

Across all steps, Apis explicitly avoids:
- motor‑level learning
- bypassing flight controllers
- delegating safety to RL
- tying correctness to rendering engines

---

## Philosophy

> **Apis is built outward from correctness.**  
> Control is engineered.  
> Learning is layered.  
> Visualization follows — never leads.
