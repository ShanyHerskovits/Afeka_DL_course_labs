import os
import matplotlib


def save_plot(fig, filename, folder="plots"):
    """
    Save a Matplotlib figure to a specified folder.

    Parameters:
    - fig: Matplotlib figure object to be saved.
    - filename: Name of the file to save the figure as (e.g., 'plot.png').
    - folder: Name of the folder to save the figure in (default is 'plots').

    Returns:
    - Full path to the saved figure.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full file path
    filepath = os.path.join(folder, filename)

    # Save the figure
    fig.savefig(filepath)

    print(f"Plot saved to {filepath}")
    return filepath
