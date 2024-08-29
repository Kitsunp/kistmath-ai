# Kistmath_AI/visualization/real_time_plotter.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RealTimePlotter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-time Learning Curve")
        self.geometry("800x600")

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.data = []

    def update_plot(self, new_data):
        self.data.append(new_data)
        if len(self.data) > 100:
            self.data.pop(0)

        self.ax.clear()
        self.ax.plot(self.data)
        self.canvas.draw()

    def update_documentation(self):
        """
        Updates the documentation file with the latest code examples.
        """
        doc_content = f"""
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

## Dependencies

- `tkinter`
- `matplotlib`

## Example Usage

```python
from Kistmath_AI.visualization.real_time_plotter import RealTimePlotter

# Initialize the plotter
plotter = RealTimePlotter()

# Simulate real-time data update
import random
import time

for _ in range(200):
    new_data = random.random()
    plotter.update_plot(new_data)
    plotter.update()
    time.sleep(0.1)
```
        """
        with open("Kistmath_AI/docs/real_time_plotter.md", "w") as doc_file:
            doc_file.write(doc_content)