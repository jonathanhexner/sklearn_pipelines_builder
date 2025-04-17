import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")  # ✅ Non-interactive, no GUI

def save_scatter_plot(y_true, y_pred, output_folder, name='', filename="scatter_plot.png"):
    """
    Saves a scatter plot of model predictions vs ground truth.

    Parameters:
    - y_true: Ground truth values (array-like)
    - y_pred: Model predictions (array-like)
    - output_folder: Folder where the plot should be saved
    - filename: Name of the output file
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title(f"{name} Model Predictions vs Ground Truth")
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()
