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