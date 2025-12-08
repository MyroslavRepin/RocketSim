# RocketSim Usage Examples

⚠️ **WARNING**: This is educational software only. Do not use for real aerospace applications.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Basic Simulation

```bash
# Default stabilization scenario
python src/main.py

# With visualization
python src/main.py --plot

# Custom duration
python src/main.py --duration 60 --dt 0.005
```

### 3. Run Demo Scenarios

```bash
# Static stabilization
python demo_scenarios/static_stabilization.py

# Disturbance rejection
python demo_scenarios/disturbance_rejection.py

# Guidance tracking
python demo_scenarios/guidance_tracking.py
```

### 4. Run Tests

```bash
python -m unittest discover tests/ -v
```

## Command Line Options

```bash
python src/main.py --help

Options:
  --duration SECONDS     Simulation duration (default: 30.0)
  --dt SECONDS          Time step (default: 0.01)
  --scenario TYPE       Scenario: stabilization, disturbance, guidance
  --initial-roll DEG    Initial roll angle (default: 0.0)
  --initial-pitch DEG   Initial pitch angle (default: 0.0)
  --initial-yaw DEG     Initial yaw angle (default: 0.0)
  --plot                Display plots after simulation
  --save-plot PATH      Save plots to file
  --animate             Show 3D animation
  --save-animation PATH Save animation as GIF
  --verbose             Print detailed progress
```

## Example Commands

```bash
# Run disturbance rejection scenario with plots
python src/main.py --scenario disturbance --plot --verbose

# Start from tilted position
python src/main.py --initial-roll 45 --initial-pitch 30 --plot

# Long simulation with fine time step
python src/main.py --duration 120 --dt 0.001 --save-plot results.png

# Guidance tracking scenario
python src/main.py --scenario guidance --duration 60 --plot
```

## Programmatic Usage

```python
import numpy as np
from src.simulation import RocketSimulation, SimulationConfig

# Create configuration
config = SimulationConfig(
    dt=0.01,
    duration=30.0,
    max_rate=2.0,
    max_torque=10.0
)

# Initialize simulation
sim = RocketSimulation(config)

# Set initial conditions
initial_euler = np.radians([30, 20, 0])  # 30° roll, 20° pitch
initial_q = sim.physics.euler_to_quaternion(*initial_euler)
sim.reset(initial_quaternion=initial_q)

# Set target
sim.set_target_attitude(euler=np.zeros(3))  # Stabilize to zero

# Run simulation
history = sim.run(verbose=True)

# Plot results
sim.plot_results(save_path="my_simulation.png")

# Access data
print(f"Final attitude: {history['euler_angles'][-1]}")
print(f"Final rates: {history['angular_velocity'][-1]}")
```

## Module Structure

```
RocketSim/
├── src/
│   ├── main.py              # CLI entry point
│   └── simulation.py        # Main simulator class
├── modules/
│   ├── physics.py           # Dynamics
│   ├── sensors.py           # IMU simulation
│   ├── estimator.py         # State estimation
│   ├── guidance.py          # Target calculation
│   ├── controllers.py       # Control laws
│   ├── control_allocation.py # Torque to elevon mapping
│   ├── aerodynamics.py      # Aero model
│   ├── safety.py            # Safety monitoring
│   └── visualization.py     # Plotting
├── demo_scenarios/          # Example scenarios
├── tests/                   # Unit tests
└── requirements.txt         # Dependencies
```

## Understanding the Output

### Simulation Results

The simulator produces:

1. **Time history plots**: Roll/pitch/yaw, angular rates, control torques, elevon deflections
2. **Safety summary**: Events, violations, warnings
3. **Statistics**: Final attitude, RMS errors, rate information

### Safety Monitor

The safety monitor logs:
- **Critical events**: Rate limit violations, unsafe conditions
- **Warnings**: Large attitude errors, saturation
- **Info**: Mode changes, state transitions

This is normal behavior and helps understand system performance.

### Typical Behavior

- Initial transients as controller engages
- Some oscillation around target (realistic for PD/PI control)
- Gradual convergence to target attitude
- Residual small errors due to sensor noise

## Customization

### Tuning Controller Gains

Edit `SimulationConfig` parameters:

```python
config = SimulationConfig(
    outer_kp=np.array([3.0, 3.0, 3.0]),  # Attitude proportional
    outer_kd=np.array([1.5, 1.5, 1.5]),  # Attitude derivative
    inner_kp=np.array([0.8, 0.8, 0.8]),  # Rate proportional
    inner_ki=np.array([0.2, 0.2, 0.2])   # Rate integral
)
```

### Changing Inertia

```python
custom_inertia = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0]
])

config = SimulationConfig(inertia=custom_inertia)
```

### Adding Custom Disturbances

```python
def my_disturbance(t):
    """Custom disturbance function"""
    if 5.0 <= t <= 10.0:
        return np.array([1.0, 0.5, 0.2])
    return np.zeros(3)

history = sim.run(external_torque_func=my_disturbance)
```

---

**Remember**: This is for educational research only. Never use for real rocket systems.
