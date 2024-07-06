import matplotlib.pyplot as plt
import io
import multiprocessing
import queue
from numpy import np
def plot_learning_curves(all_history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    stages = ['basic', 'algebra', 'precalculus', 'calculus']
    
    for i, stage in enumerate(stages):
        ax = axes[i // 2, i % 2]
        stage_history = next(h for h in all_history if h['stage'] == stage)
        
        losses = []
        maes = []
        for history in stage_history['fold_histories']:
            losses.extend(history['loss'])
            maes.extend(history['mae'])
        
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label='Loss')
        ax.plot(epochs, maes, label='MAE')
        
        ax.set_title(f'{stage.capitalize()} Stage')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def real_time_plotter(plot_queue):
    plt.switch_backend('agg')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs, losses, maes = [], [], []
    
    while True:
        try:
            data = plot_queue.get(timeout=1)
            if data is None:
                break
            
            epochs.append(data['epoch'])
            losses.append(data['loss'])
            maes.append(data['mae'])
            
            ax1.clear()
            ax2.clear()
            
            ax1.plot(epochs, losses, 'b-', label='Loss')
            ax2.plot(epochs, maes, 'g-', label='MAE')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            
            ax1.set_title(f"Epoch {data['epoch']}/{data['total_epochs']}")
            
            for i, example in enumerate(data['examples']):
                true_value = example['true']
                predicted_value = example['predicted']
                
                if isinstance(true_value, np.ndarray):
                    true_str = f"[{', '.join(f'{v:.4f}' for v in true_value)}]"
                else:
                    true_str = f"{true_value:.4f}"
                
                if isinstance(predicted_value, np.ndarray):
                    pred_str = f"[{', '.join(f'{v:.4f}' for v in predicted_value)}]"
                else:
                    pred_str = f"{predicted_value:.4f}"
                
                ax2.text(0.05, 0.9 - i*0.15, 
                         f"Problem: {example['problem']}\nTrue: {true_str}, Predicted: {pred_str}", 
                         transform=ax2.transAxes, fontsize=8, verticalalignment='top')
            
            fig.canvas.draw()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            with open(f'training_progress_epoch_{data["epoch"]}.png', 'wb') as f:
                f.write(buf.getvalue())
            
        except queue.Empty:
            pass
    
    plt.close(fig)