# Apis --- Autonomous Perception and Intelligent Systems

## Overview

**Apis** is a research and engineering program focused on building
autonomous robotic systems that can safely and reliably operate in the
real world.

The core philosophy of Apis is:

> Intelligence should be learned and expressed at the **command /
> decision interface**, not at the raw actuator level.

Rather than requiring learning systems to discover both: - how to
control complex hardware - and how to perform tasks

Apis separates these concerns through layered learning and modeling.

------------------------------------------------------------------------

## Core Design Philosophy

### Learning Above Control

Apis assumes that low-level stabilization and actuation are solved by
dedicated controllers (PID, cascaded PID, LQR, etc.).

Learning systems operate **above** this layer, issuing high-level
commands such as: - move in direction - move to location - change
velocity - change heading - land / takeoff

This mirrors real-world systems such as: - modern aircraft fly-by-wire -
autonomous vehicle drive-by-wire - industrial robotics motion planners

------------------------------------------------------------------------

### Command-Level Modeling

Apis focuses on learning models of how systems behave **when
commanded**, not how motors generate forces.

We model:

    (current state, command) → next state

This represents the behavior of the full closed-loop system: - onboard
controller - actuator dynamics - physical system - estimator behavior

This is referred to throughout Apis as:

**Closed-loop behavior modeling**\
(or simply: closed-loop models)

------------------------------------------------------------------------

## Why This Matters

Requiring learning systems to operate at motor signal level forces them
to learn two problems simultaneously:

1.  Task execution\
2.  Hardware stabilization

This dramatically increases sample complexity and instability.

Apis instead enables learning systems to behave like **pilots**, not
motor drivers.

------------------------------------------------------------------------

## System Architecture

Apis is structured as a staged learning pipeline.

------------------------------------------------------------------------

# Step 1 --- Nuc

## Closed-Loop Behavior Data Generation

### Purpose

Generate high-quality trajectory datasets from controlled systems.

### Role

Nuc acts as a **behavior generator** for closed-loop systems.

It produces datasets of the form:

    (state_t, command_t, state_{t+1})

### Data Sources

Nuc can operate with:

-   Fully transparent controllers (toy_controller, simple_controller)
-   Known simulation environments
-   Real hardware
-   Black-box systems (later phase)

### Why Nuc Exists

Before learning surrogate models, we must understand:

-   What excitation produces informative data
-   How controller internal state appears in behavior
-   How dataset coverage affects rollout stability
-   How command distributions shape learnability

------------------------------------------------------------------------

# Step 2 --- Brood

## Closed-Loop Behavior Model Learning

### Purpose

Learn surrogate models that reproduce closed-loop system behavior.

### Role

Brood learns transition models that can replace real systems in
simulation.

### Learns

    f(state, command) → next state

Optionally:

    f(state history, command history) → next state

to handle hidden internal controller state.

### Important Clarification

Brood does **not** attempt to reconstruct controller internals.\
It learns the observable command → behavior mapping.

------------------------------------------------------------------------

# Why We Start With Transparent Controllers

Transparent controllers provide:

-   Ground truth behavior
-   Known internal structure
-   Controlled experimentation
-   Debuggable failure modes

This allows us to learn:

-   Dataset design principles
-   Model class tradeoffs
-   Rollout stability requirements
-   When memory is required
-   How closed-loop dynamics hide internal structure

------------------------------------------------------------------------

# Step 3 --- Waggle

## Unity-Based Controller-in-the-Loop Scaffolding

Waggle introduces structured interaction between deterministic control
logic and a real-time 3D environment.

Goal: - Use Unity to send high-level actions - Receive state estimates
over fixed time windows - Evaluate timing, latency, and action semantics

Unity acts as orchestrator and viewer, never as the source of truth.

------------------------------------------------------------------------

# Step 4 --- Propolis

## Isaac Sim Integration

Propolis extends the same controller-in-the-loop architecture to **Isaac
Sim**.

Goal: - Operate a virtual Crazyflie controller inside Isaac Sim -
Compare behavior, timing, and sensing with Unity-based workflows

This step focuses on higher-fidelity physics and robotics-oriented
simulation.

------------------------------------------------------------------------

# Step 5 --- Scopa

## Virtual Sensor Expansion & Data Generation

Scopa introduces richer virtual sensing.

Goal: - Add sensors such as RGB cameras, depth, or LiDAR - Generate
aligned datasets combining controller state, actions, and sensor
observations

Unity and Isaac Sim become data generators rather than control
authorities.

------------------------------------------------------------------------

# Step 6 --- Proboscis

## Perception, Fusion, and Active Control Pipelines

Proboscis closes the loop from perception to action.

Goal: - Build pipelines that connect sensor data to high-level commands
using: - SLAM / localization - sensor fusion - learned or classical
policies - active perception strategies

This is where full autonomy emerges as a system.

------------------------------------------------------------------------

## Safety and Non-Goals

Across all steps, Apis explicitly avoids: - motor-level learning -
bypassing flight controllers - delegating safety to RL - tying
correctness to rendering engines

------------------------------------------------------------------------

## Philosophy

> **Apis is built outward from correctness.**\
> Control is engineered.\
> Learning is layered.\
> Visualization follows --- never leads.

------------------------------------------------------------------------

# What Defines Success

A successful closed-loop surrogate model:

-   Produces stable multi-step rollouts
-   Matches real system response to commands
-   Preserves timing, lag, overshoot characteristics
-   Works inside RL training loops
-   Generalizes across command distributions

Not required: - Recovery of true physical parameters - Motor-level
interpretability - Internal controller reconstruction

------------------------------------------------------------------------

# Core Contract

> If the surrogate receives the same command in the same state as the
> real system, it should produce the same next-state distribution.

------------------------------------------------------------------------

# Educational Value

Apis is also a learning platform for:

-   Closed-loop system identification
-   Dataset design for dynamical systems
-   Rollout stability vs one-step accuracy
-   Hidden state inference
-   World model learning
-   Sim-to-real interface design

------------------------------------------------------------------------

# Terminology

  Term                 Meaning
  -------------------- --------------------------------------------------
  Controller           Code that stabilizes system and drives actuators
  Closed-loop system   Controller + actuators + physics + estimator
  Closed-loop model    Learned command → behavior model
  Surrogate            Learned stand-in for real system behavior

------------------------------------------------------------------------

# Executive Summary

Apis develops methods for learning surrogate models that reproduce how
real systems respond to commands. These surrogates allow autonomous
behavior training in simulation without requiring access to internal
controllers or motor-level physics.

The program progresses from transparent controller simulation (Nuc,
Brood) through controller-in-the-loop simulation environments (Waggle,
Propolis), to sensor-driven autonomy and perception-integrated control
(Scopa, Proboscis).
