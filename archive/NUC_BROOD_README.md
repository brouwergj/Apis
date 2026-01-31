# Nuc & Brood --- Project Intent and Conceptual Model

## 1. High-Level Purpose

The long-term goal of this project is to enable training and evaluation
of autonomous drone behaviors **in simulation**, such that learned
policies transfer to real drones that use onboard controllers (which may
be partially or fully black-box).

To achieve this, we do **not** attempt to model low-level physics or
motor-level control directly.\
Instead, we model how a drone behaves **when commanded through its
normal control interface**.

The core idea is:

> Learn models that reproduce how a real drone responds to commands, not
> how motors produce thrust.

------------------------------------------------------------------------

## 2. The Key Conceptual Shift

### ❌ Not the current goal

Recovering true physical plant parameters: - Mass - Drag - Motor
dynamics - Aerodynamic coefficients

(Classical plant identification)

### ✅ Actual goal

Learn:

    (current state, command) → next state

through the drone's command interface.

Equivalent terms: - Closed-loop dynamics - Command-level behavior
model - Pilot-level response model - Closed-loop world model

------------------------------------------------------------------------

## 3. Pilot Analogy

The RL agent is treated as a **pilot**, not a motor controller.

Like modern aircraft: - Pilot: "go this way" - Flight computer decides
how - Aircraft + controller produce motion

We model: \> Drone + onboard controller + physics + estimator\
as one composite system.

------------------------------------------------------------------------

## 4. Why This Matters for RL

Instead of learning: - Task AND - Low-level flight control

We want RL to learn:

    sensors → high-level action

Examples: - velocity commands - position deltas - joystick-style
commands - go-to / move / turn primitives

The surrogate model handles physical execution.

------------------------------------------------------------------------

## 5. Project Architecture

### Nuc --- Data Generation

Purpose: - Generate trajectory data from controllers or hardware - Probe
system response using command sequences

Produces:

    (state_t, command_t, state_{t+1})

Works with: - Transparent controllers - Real hardware - Black-box
systems

------------------------------------------------------------------------

### Brood --- Model Learning

Purpose: - Train surrogate transition models - Enable high-fidelity
simulation

Learns:

    f(s_t, command_t) → s_{t+1}

Optional extension:

    f(history, command) → next state

------------------------------------------------------------------------

## 6. Why Start With Transparent Controllers

We first learn: - Dataset design - Loss design - Rollout stability -
Data coverage needs - Hidden state effects

Tools: - toy_controller - simple_controller - simulated physics

------------------------------------------------------------------------

## 7. What We Are Learning (Technical)

> Stable rollout-capable transition models of closed-loop controlled
> dynamical systems using command-level interfaces.

Short version: \> Closed-loop dynamics models.

------------------------------------------------------------------------

## 8. What "Good" Looks Like

A good surrogate: - Stable multi-step rollouts - Matches real response
to commands - Preserves delay / lag / overshoot - Works inside RL
simulation loops

Not required: - True physical parameter recovery - Motor-level
interpretability - Internal controller reconstruction

------------------------------------------------------------------------

## 9. Long-Term Path

### Phase 1 --- Transparent Teachers

(Current phase)

Learn: - Modeling methodology - Failure modes - Data requirements

------------------------------------------------------------------------

### Phase 2 --- Partial Unknowns

Learn: - Hidden state handling - Latent state modeling - History-based
prediction

------------------------------------------------------------------------

### Phase 3 --- True Black Boxes

Apply pipeline to real drones.

------------------------------------------------------------------------

## 10. Terminology

Preferred: - Closed-loop dynamics - CL model - Command-level model -
Surrogate

Avoid confusion between: - Controller (decision maker) - Surrogate
(behavior emulator)

------------------------------------------------------------------------

## 11. Core Contract

> If the surrogate receives the same command in the same state as the
> real drone, it should produce the same next state distribution.

------------------------------------------------------------------------

## 12. Educational Value

This project teaches: - Closed-loop system identification - Dataset
design for dynamical systems - Rollout stability vs one-step accuracy -
Hidden state modeling - World model training - Sim-to-real interface
design

------------------------------------------------------------------------

# Executive Summary

We are building a pipeline to learn surrogate models that reproduce how
drones respond to commands.\
These models enable training autonomous behaviors in simulation without
needing access to onboard controller internals or motor-level physics.

We start with transparent controllers, then scale to black-box
real-world systems.
