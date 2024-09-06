# Kistmath_AI/visualization/plotting.py

This file contains functions for plotting learning curves and real-time plotting.

## Functions

### plot_learning_curves(all_history)

- **Description**: Plots learning curves for all stages of the curriculum.
- **Parameters**:
  - `all_history`: A list of dictionaries containing training history for each stage.

#### Example
```python
from Kistmath_AI.visualization.plotting import plot_learning_curves

# Example training history
all_history = [
    {"loss": [0.1, 0.05, 0.02], "val_loss": [0.15, 0.1, 0.05]},
    {"loss": [0.09, 0.04, 0.01], "val_loss": [0.14, 0.09, 0.04]}
]

# Plot the learning curves
plot_learning_curves(all_history)
```

### real_time_plotter(plot_queue)

- **Description**: Plots real-time training progress.
- **Parameters**:
  - `plot_queue`: A multiprocessing Queue containing plot data.

#### Example
```python
from Kistmath_AI.visualization.plotting import real_time_plotter
import multiprocessing

# Initialize a multiprocessing queue
plot_queue = multiprocessing.Queue()

# Example data
plot_queue.put([1, 2, 3, 4, 5])

# Plot the real-time training progress
real_time_plotter(plot_queue)
```

## Dependencies

- `matplotlib`
- `io`
- `multiprocessing`
- `queue`
- `numpy`