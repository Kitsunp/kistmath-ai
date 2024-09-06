import multiprocessing
from visualization.plotting import real_time_plotter
from utils.logging import log_message

def start_visualization_process():
    plot_queue = multiprocessing.Queue()
    plot_process = multiprocessing.Process(target=real_time_plotter, args=(plot_queue,))
    plot_process.start()
    log_message("Proceso de visualización en tiempo real iniciado")
    return plot_queue, plot_process

def stop_visualization_process(plot_queue, plot_process):
    plot_queue.put(None)
    plot_process.join()
    log_message("Proceso de visualización finalizado")