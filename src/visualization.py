import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


def visualize_distributions(original_data, synthetic_data, column, iteration, kl_divergences, visualization_mode, ax):
    # Print KL divergences before visualization
    for i, kl_div in enumerate(kl_divergences, start=1):
        print(f"Iteration {i} - KL Divergence: {kl_div}")
    
    # Create a gaussian kde for the original and synthetic data
    kde_original = gaussian_kde(original_data)
    kde_synthetic = gaussian_kde(synthetic_data)
    x_eval = np.linspace(min(original_data.min(), synthetic_data.min()), 
                         max(original_data.max(), synthetic_data.max()), 500)
    y_original = kde_original(x_eval)
    y_synthetic = kde_synthetic(x_eval)
    
    # Plot original data distribution
    ax.plot(x_eval, y_original, label='Original Data', color='black', linestyle="--", linewidth=2)
    
    # Calculate the distance and normalize it
    distance = np.abs(y_synthetic - y_original)
    normalized_distance = distance / max(distance)

    # Initialize line width and colors
    line_width = np.full_like(normalized_distance, 3)  # Default line width
    colors = np.full((len(normalized_distance), 3), [0, 1, 0])  # Default color (green)

    # Modify line width and colors based on visualization mode
    if visualization_mode == "line":
        # Varying line width, constant green color
        line_width = 2 + (1 - normalized_distance) * 4
    elif visualization_mode == "color":
        # Constant line width, varying color from red to green
        red_component = np.clip(normalized_distance, 0.2, 1)
        green_component = np.clip(1 - normalized_distance, 0.2, 1)
        colors = np.vstack((red_component, green_component, np.zeros_like(red_component))).T
    elif visualization_mode == "both":
        # Varying line width and color
        line_width = 2 + (1 - normalized_distance) * 4
        red_component = np.clip(normalized_distance, 0.2, 1)
        green_component = np.clip(1 - normalized_distance, 0.2, 1)
        colors = np.vstack((red_component, green_component, np.zeros_like(red_component))).T

    # Create a multicolored line for the synthetic data distribution
    points = np.array([x_eval, y_synthetic]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Creating a LineCollection for the synthetic data distribution
    lc = LineCollection(segments, linewidths=line_width, colors=colors)
    ax.add_collection(lc)

    # Set plot limits, labels, titles, and legends
    ax.set_xlim(x_eval.min(), x_eval.max())
    ax.set_ylim(min(y_synthetic.min(), y_original.min()), max(y_synthetic.max(), y_original.max()))
    ax.set_title(f'Distribution of {column} - Iteration {iteration}')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    if visualization_mode in ["color", "both"]:
        # Create a color bar for the 'color' and 'both' modes
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Synthetic Data Distribution Proximity to Original Data(Green:Close, Red:Far)')
        ax.legend(handles=[Line2D([0], [0], color='black', label='Original Data Distribution', linestyle="--", linewidth=2)], loc='upper right')
    else:
        # Use simple legends for the 'line' mode
        legend_elements = [
            Line2D([0], [0], color='black', label='Original Data Distribution', linestyle="--", linewidth=2),
            Line2D([0], [0], color='green', label='Synthetic Data Distribution')
        ]
        ax.legend(handles=legend_elements)