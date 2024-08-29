# Kistmath_AI/visualization/plotting.py

This file contains functions for plotting learning curves and real-time plotting.

## Functions

### plot_learning_curves(all_history)

- **Description**: Plots learning curves for all stages of the curriculum.
- **Parameters**:
  - `all_history`: A list of dictionaries containing training history for each stage.

### real_time_plotter(plot_queue)

- **Description**: Plots real-time training progress.
- **Parameters**:
  - `plot_queue`: A multiprocessing Queue containing plot data.

## Dependencies

- `matplotlib`
- `io`
- `multiprocessing`
- `queue`
- `numpy`