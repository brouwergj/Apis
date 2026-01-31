# Nuc --- Deterministic Control & Closed-Loop Behavior Data Core (Apis)

This document is both:

• The **technical reference** for the Nuc subsystem\
• The **conceptual foundation** for how Apis generates closed-loop
behavioral data

------------------------------------------------------------------------

## Relationship to Apis

-   **Apis** defines the full autonomy and learning roadmap.\
-   **Nuc** is the deterministic behavioral data engine.\
-   **Brood** will learn models from Nuc-generated data.

In Apis terms:

  Layer               Role
  ------------------- -----------------------------------------------
  Flight Controller   Stabilizes hardware
  Nuc                 Generates closed-loop behavioral trajectories
  Brood               Learns surrogate closed-loop behavior models
  Waggle+             Embeds behavior into interactive environments

------------------------------------------------------------------------

# Core Concept: What Nuc Actually Produces

Nuc generates **closed-loop behavioral trajectories**, not physics
ground truth, and not controller internals.

Formally:

    (state_t, command_t) → state_{t+1}

Where the system includes:

-   Controller logic\
-   Actuator dynamics\
-   Physical system\
-   Estimator effects\
-   Timing and latency behavior

This is called **Closed-Loop Behavior Modeling** across Apis.

------------------------------------------------------------------------

# Core Design Philosophy

## Learning Above Control

Apis assumes:

Low-level stabilization is solved by engineered controllers:

-   PID\
-   Cascaded PID\
-   LQR\
-   MPC\
-   Flight firmware stacks

Learning systems operate above this layer, issuing pilot-like commands:

-   Move direction\
-   Move to location\
-   Velocity change\
-   Heading change\
-   Takeoff / land

This mirrors real-world fly-by-wire systems.

------------------------------------------------------------------------

## Command-Level Modeling

Nuc exists because we do **not** want learning systems to learn
motor-level control.

Instead, we want them to learn task behavior at the command interface.

Why?

If learning includes motor control: - sample complexity explodes\
- training instability increases\
- safety guarantees disappear\
- sim-to-real becomes fragile

Nuc allows Apis to learn at the **pilot abstraction level**.

------------------------------------------------------------------------

# Why Nuc Exists (From a Learning Perspective)

Before training surrogate models, we must learn:

• What command sequences reveal hidden controller state\
• How excitation influences identifiability\
• How dataset coverage influences rollout stability\
• How command distributions influence learnability

Transparent controllers are used first because they provide ground truth
behavior.

------------------------------------------------------------------------

# The Role of Determinism

Determinism is not an implementation detail.\
It is a **data integrity guarantee**.

Nuc enforces:

• Fixed timestep\
• Seeded randomness\
• Single authority simulation loop\
• Log-first architecture

This ensures:

Dataset reproducibility\
Debuggable learning failures\
Comparable training experiments

------------------------------------------------------------------------

# What We Are Building (System View)

Nuc is a system for:

• Running deterministic controller simulations in Python\
• Visualizing runs in Unity\
• Generating replayable trajectory logs

Key principles:

-   Deterministic simulation lives in Python\
-   Unity orchestrates and visualizes\
-   Logs are the primary artifact\
-   Replay is log-driven

------------------------------------------------------------------------

# Process Architecture (Unity ↔ Nuc)

Nuc runs as a **separate Python process** providing:

1)  WebSocket control plane (Unity → Python)\
2)  Optional UDP telemetry (Python → Unity, visualization only)

Key architectural truth:

Unity is never the simulation authority.\
Python is the clock, truth, and dataset generator.

------------------------------------------------------------------------

# RunConfig Contract

RunConfigs define:

• Episode timing\
• Controller selection\
• Reference programs\
• Randomization\
• Logging behavior\
• Telemetry behavior

RunConfigs are versioned experimental definitions.

They are critical for:

Reproducibility\
Dataset lineage\
Experiment traceability

------------------------------------------------------------------------

# Files and Responsibilities

## Python (Nuc)

### runner.py

Episode orchestration and controller dispatch.

### Controller Implementations

-   toy_controller --- deterministic quad model + cascaded control\
-   crazyflie_controller --- firmware-style PID supervision\
-   simple_controller --- minimal baseline controller

These exist to:

• Generate known-behavior datasets\
• Study identifiability limits\
• Study hidden state effects

------------------------------------------------------------------------

## Unity (Orchestrator + Viewer)

Unity is:

• UI\
• Process orchestrator\
• Visualization endpoint\
• Replay engine

Unity is NOT:

• Physics authority\
• Timing authority\
• Dataset source

------------------------------------------------------------------------

# Data Layout

Logs and runconfigs live outside Unity to guarantee:

• Tool independence\
• Dataset longevity\
• Training pipeline portability

------------------------------------------------------------------------

# Runtime Modes

## Single Run

Manual dataset generation.

## Batch Run

Dataset sweep across command programs.

## Replay

Ground truth playback independent of runtime.

Replay is log-truth, not simulation-truth.

------------------------------------------------------------------------

# Python Runtime Guarantees

The Python runtime guarantees:

• Single START message per run\
• Deterministic fixed dt stepping\
• Log-first execution\
• Optional telemetry mirror\
• Clean DONE + exit semantics

This supports batch experiment automation.

------------------------------------------------------------------------

# Unity Runtime Guarantees

Unity guarantees:

• Clean process lifecycle management\
• Safe WebSocket orchestration\
• Latest-only telemetry display\
• Clean replay mode separation

------------------------------------------------------------------------

# What Success Looks Like (For Nuc)

Nuc is successful when:

• Generated datasets allow stable surrogate training\
• Closed-loop rollout behavior is learnable\
• Dataset variation reveals hidden dynamics\
• Replay matches runtime ground truth

------------------------------------------------------------------------

# What Nuc Explicitly Does NOT Do

Nuc does not:

• Learn controllers\
• Learn motor control\
• Replace flight firmware\
• Perform RL training\
• Perform perception

Nuc generates **behavioral truth datasets**.

------------------------------------------------------------------------

# Nuc → Brood Contract

Brood assumes:

If Nuc generates sufficient data coverage,\
Brood can learn a closed-loop surrogate model.

------------------------------------------------------------------------

# Educational Value

Nuc enables controlled experiments in:

Closed-loop system identification\
Excitation design\
Dataset sufficiency analysis\
Hidden state detectability\
Rollout stability testing

------------------------------------------------------------------------

# Terminology

  Term                   Meaning
  ---------------------- ---------------------------------------------
  Controller             Stabilizes system, drives actuators
  Closed-loop system     Controller + hardware + physics + estimator
  Closed-loop behavior   Observable command → state evolution
  Nuc Dataset            Closed-loop trajectory dataset

------------------------------------------------------------------------

# One-Line Summary

Nuc is the deterministic closed-loop behavioral data engine that enables
surrogate model learning across the Apis autonomy stack.
