# Kistmath_AI/visualization/real_time_plotter.py

This file contains the implementation of the `RealTimePlotter` class, which is used to create a real-time plot of learning curves using Tkinter and Matplotlib.

## Classes

### RealTimePlotter

- **Description**: A class that extends `tk.Tk` to create a real-time plot of learning curves.
- **Methods**:
  - `__init__(self)`: Initializes the real-time plotter window.
    - **Parameters**: None
    - **Returns**: None
    - **Description**: Sets up the Tkinter window, initializes the Matplotlib figure and canvas, and prepares an empty data list for plotting.
  - `update_plot(self, new_data)`: Updates the plot with new data.
    - **Parameters**:
      - `new_data`: The new data point to be added to the plot.
    - **Returns**: None
    - **Description**: Appends the new data to the internal data list, removes the oldest data point if the list exceeds 100 points, clears the current plot, plots the updated data list, and redraws the canvas.

#### Example
```python
from Kistmath_AI.visualization.real_time_plotter import RealTimePlotter

# Initialize the real-time plotter
plotter = RealTimePlotter()

# Example data
new_data = [1, 2, 3, 4, 5]

# Update the plot with new data
plotter.update_plot(new_data)
```

## Dependencies

- `tkinter`
- `matplotlib`