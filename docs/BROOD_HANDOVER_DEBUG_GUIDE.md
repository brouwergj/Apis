# Handover: Debugging Brood Closed-Loop Behavioral Surrogate (End-to-End)

## Goal

Produce a surrogate model that predicts next state with high precision
under closed-loop command semantics.

Target mapping (behavioral):

(s_t, c_t) → s\_{t+1}

Where the system implicitly includes controller + inner-loop lags +
physics + estimator effects.

------------------------------------------------------------------------

## 1. Pipeline Contract Verification

### 1.1 Log ↔ RunConfig Naming Contract

Estimator expects:

Logs: ep\_`<name>`{=html}\_YYYYMMDD_HHMMSS.csv

RunConfigs: ep\_`<name>`{=html}.json

Current runner writes: {run_id}*{controller}*{timestamp}.csv

Likely failure: RunConfig not found → estimator silently falls back to
default dynamics.

Required fixes (choose one): • Change runner naming (preferred) • Update
run_id parsing logic • Add multi-attempt lookup in estimator

Verification: Estimator must produce ZERO missing runconfig warnings.

------------------------------------------------------------------------

### 1.2 RunConfig Schema Alignment

Runner reads: controller parameters dynamics block (optional, defaults
otherwise)

Estimator reads: dynamics: tau_roll, tau_pitch, tau_thrust, wind
controller: tilt/yaw/thrust limits

Likely failure: Estimator assumes different dynamics than runner.

Fix: Ensure RunConfig contains full dynamics block OR log dynamics into
CSV.

------------------------------------------------------------------------

### 1.3 Command Semantics

Dataset contains: ref = position/yaw reference u = controller outputs
(roll, pitch, yaw_rate, thrust)

Estimator currently learns controller surrogate, not behavioral
surrogate.

Decision required: Option A: Controller surrogate → predict u Option B:
Behavioral surrogate → predict next state directly

For next-state precision, Option B is recommended.

------------------------------------------------------------------------

## 2. Data Sanity Checks

### 2.1 Step Alignment

Runner logs post-step state.

Risk: Off-by-one learning target.

Fix: Shift dataset OR change runner logging to pre-step state.

------------------------------------------------------------------------

### 2.2 Thrust Lag Modeling

Estimator assumes thrust_actual latent state. Logs only contain
thrust_cmd.

Risk: Double-lag or incorrect lag modeling.

Fix (preferred): Log thrust_actual from simulator.

------------------------------------------------------------------------

## 3. Objective Function Mismatch

### Current Rollout Loss

Optimizes tracking reference trajectory.

### Required Rollout Loss

Must optimize matching logged behavior trajectory.

Fix: Compare predicted trajectory against logged states, not reference
signals.

------------------------------------------------------------------------

## 4. Recommended Debug Order

Step A --- RunConfig Matching Step B --- dt + Alignment Verification
Step C --- Rollout Loss Target Correction Step D --- Baseline Dynamics
Reproduction Test Step E --- Explicit Modeling Target Selection

------------------------------------------------------------------------

## 5. Hidden State Considerations

Simple controller includes slew limiting (hidden internal state).

Mitigations: • Include previous control input • Use history window • Use
latent state / RNN • Or switch to direct behavioral transition model

------------------------------------------------------------------------

## 6. Distribution Coverage

Training coverage must include: • Saturations • Aggressive yaw • Z
boundary transitions • Hover micro-corrections

------------------------------------------------------------------------

## 7. Precision Evaluation Metrics

Required metrics:

One-step prediction error K-step rollout error curves Closed-loop
stability Frequency response / lag characteristics

------------------------------------------------------------------------

## 8. Highest Probability Failure Sources

1.  RunConfig mismatch due to filename parsing
2.  Rollout loss targeting reference instead of logged trajectory
3.  State alignment off-by-one
4.  Hidden controller state not modeled
5.  Dynamics mismatch between simulator and estimator

------------------------------------------------------------------------

## 9. Required Debug Artifacts

Assistant must produce:

Data audit report Contract matching report Baseline dynamics
reproduction test Updated estimator rollout loss implementation
Precision + rollout stability evaluation report

------------------------------------------------------------------------

## Final Principle

We are fitting closed-loop behavioral dynamics, not controller code.

If same state + same command → same next-state distribution, the
surrogate is correct.
