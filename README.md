# RocketSim - Educational Rocket Stabilization Simulator

## ⚠️ CRITICAL SAFETY WARNING ⚠️

**ВНИМАНИЕ! Данный код предназначен только для симуляции и исследования алгоритмов стабилизации. Любое практическое применение кода, а также его использование для реального конструирования, испытаний или эксплуатации летательных аппаратов категорически запрещено и может привести к опасным последствиям.**

**WARNING! This code is intended ONLY for simulation and research of stabilization algorithms. Any practical application of this code, as well as its use for actual construction, testing, or operation of aircraft is strictly prohibited and may lead to dangerous consequences.**

This is a purely educational software simulator. Do not use in any real-world aerospace applications.

---

## Overview

RocketSim is a modular Python-based simulator for rocket attitude stabilization and control. It implements a complete control pipeline from physics simulation through sensor modeling, state estimation, guidance, control, and visualization.

## Architecture

The simulator consists of the following key modules:

### 1. Physics Module (`modules/physics.py`)
- **Purpose**: Models 3-axis rotational dynamics of the rocket
- **Implementation**: Uses quaternions for attitude representation to avoid gimbal lock
- **Dynamics**: Integrates Euler's rotational equations: `I·ω̇ = τ - ω × (I·ω)`
- **Rationale**: Quaternions chosen over Euler angles for numerical stability and singularity-free representation

### 2. Sensors Module (`modules/sensors.py`)
- **Purpose**: Simulates IMU sensor outputs with realistic noise characteristics
- **Components**:
  - Gyroscope: measures angular rates with white noise and bias drift
  - Accelerometer: measures specific force with noise
  - Optional: GPS, airspeed sensor
- **Rationale**: Realistic sensor modeling is essential for testing estimator robustness

### 3. State Estimator Module (`modules/estimator.py`)
- **Purpose**: Estimates attitude and angular rates from noisy sensor data
- **Algorithms**:
  - **Complementary Filter** (default): Simple, computationally efficient, good for well-characterized sensors
  - **Extended Kalman Filter** (optional): More accurate with proper noise modeling, but more complex
- **Rationale**: Complementary filter chosen as default for simplicity and low computational cost; EKF available for advanced scenarios requiring optimal fusion

### 4. Guidance Module (`modules/guidance.py`)
- **Purpose**: Calculates desired attitude for mission objectives
- **Modes**:
  - Direct attitude command (quaternion or Euler angles)
  - Point tracking (calculates attitude to point rocket toward target)
- **Output**: Target quaternion for controller

### 5. Controllers Module (`modules/controllers.py`)
- **Purpose**: Dual-loop control architecture
- **Structure**:
  - **Outer Loop**: Attitude error → desired angular rates (PD controller)
  - **Inner Loop**: Rate error → control torques (PI controller)
- **Rationale**: PD chosen for outer loop (no integral needed for orientation tracking); PI for inner loop (integral removes steady-state rate errors). Two-loop structure separates slow attitude dynamics from fast rate dynamics, improving stability
- **Optional**: LQR controller available for optimal control (requires state-space model tuning)

### 6. Control Allocation Module (`modules/control_allocation.py`)
- **Purpose**: Maps control torques to 4 elevon deflections
- **Implementation**:
  - Uses effectiveness matrix to solve: `B·δ = τ`
  - Applies saturation limits (physical deflection constraints)
  - Models servo rate limits
- **Rationale**: Pseudo-inverse allocation handles redundancy in 4-elevon configuration

### 7. Aerodynamics Module (`modules/aerodynamics.py`)
- **Purpose**: Models aerodynamic torques from control surfaces
- **Implementation**:
  - Torque coefficients scaled by dynamic pressure: `q = 0.5·ρ·V²`
  - Gain scheduling: effectiveness varies with q
- **Rationale**: Simple model captures essential dependency on flight conditions

### 8. Safety Module (`modules/safety.py`)
- **Purpose**: Enforces operational constraints and fallback behaviors
- **Features**:
  - Control saturation monitoring
  - Angular rate limiting with automatic damping mode
  - Anomaly logging
- **Rationale**: Essential for preventing unrealistic behavior and studying failure modes

### 9. Visualization Module (`modules/visualization.py`)
- **Purpose**: Plots and 3D animation for analysis
- **Outputs**:
  - Time histories: roll/pitch/yaw, angular rates, control commands
  - 3D animated rocket orientation
- **Tools**: matplotlib for 2D plots, matplotlib 3D for orientation display

## Installation

```bash
# Clone repository
git clone https://github.com/MyroslavRepin/RocketSim.git
cd RocketSim

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Run default stabilization scenario
python src/main.py

# Run specific scenario
python src/main.py --scenario disturbance

# Run with custom parameters
python src/main.py --duration 60 --dt 0.01
```

### Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook demo_scenarios/demo.ipynb
```

## Demo Scenarios

Located in `demo_scenarios/`:

1. **Static Stabilization** (`static_stabilization.py`): Rocket maintains zero orientation
2. **Disturbance Rejection** (`disturbance_rejection.py`): Rocket recovers from external torque pulse
3. **Guidance Tracking** (`guidance_tracking.py`): Rocket tracks moving target point

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_estimator.py
```

## Project Structure

```
RocketSim/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── main.py              # CLI entry point
│   └── simulation.py        # Main simulator class
├── modules/
│   ├── physics.py           # Dynamics model
│   ├── sensors.py           # IMU simulation
│   ├── estimator.py         # State estimation
│   ├── guidance.py          # Target calculation
│   ├── controllers.py       # Control laws
│   ├── control_allocation.py # Torque to elevon mapping
│   ├── aerodynamics.py      # Aero forces/torques
│   ├── safety.py            # Constraints & monitoring
│   └── visualization.py     # Plotting & animation
├── demo_scenarios/
│   ├── static_stabilization.py
│   ├── disturbance_rejection.py
│   └── guidance_tracking.py
└── tests/
    ├── test_estimator.py
    ├── test_controller.py
    └── test_physics.py
```

## Limitations

- Simplified aerodynamic model (linear effectiveness, no coupling)
- No translational dynamics (position/velocity not modeled)
- Fixed inertia tensor (no fuel consumption effects)
- Atmospheric model simplified (constant density at altitude)
- No structural flexibility or propellant sloshing

## Design Decisions Rationale

### Why Quaternions?
Quaternions avoid gimbal lock, have no singularities, and provide efficient and stable numerical integration compared to Euler angles.

### Why Complementary Filter as Default?
For well-characterized sensors, the complementary filter provides excellent results with minimal computational cost. The simple high-pass + low-pass structure is easy to tune and understand.

### Why PD instead of PID for Attitude?
Integral action on orientation can cause windup during saturation. The steady-state attitude error is typically near zero if rate control is accurate, making integral action unnecessary.

### Why Dual-Loop Control?
Separating slow attitude dynamics from fast rate dynamics allows independent tuning and better performance. The inner loop stabilizes angular rates quickly, while the outer loop achieves desired orientation.

## Contributing

This is an educational project. Contributions should maintain clarity and educational value.

## License

MIT License - See LICENSE file for details.

## References

1. Stevens, B. L., & Lewis, F. L. (2003). Aircraft Control and Simulation.
2. Wertz, J. R. (Ed.). (1978). Spacecraft Attitude Determination and Control.
3. Mahony, R., Hamel, T., & Pflimlin, J. M. (2008). Nonlinear complementary filters on the special orthogonal group. IEEE TAC.

---

## ⚠️ FINAL REMINDER ⚠️

**This software is for educational simulation only. Do not apply to real aerospace systems.**