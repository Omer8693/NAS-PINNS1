import numpy as np
import matplotlib.pyplot as plt

from .common import finalize_plot

SIMPLE_CMAP = "YlGnBu"


def plot_poisson_results(
    model,
    predict_on_grid_fn,
    save_path,
    pred_title,
    suptitle=None,
):
    rel_l2, x_np, y_np, phi_pred, phi_exact = predict_on_grid_fn(model, test_nx=150, test_ny=150)

    abs_err = np.abs(phi_pred - phi_exact)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    cs0 = axes[0].contourf(x_np, y_np, phi_exact, levels=50, cmap=SIMPLE_CMAP)
    axes[0].set_title("Exact φ(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(x_np, y_np, phi_pred, levels=50, cmap=SIMPLE_CMAP)
    axes[1].set_title(pred_title)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(x_np, y_np, abs_err, levels=50, cmap=SIMPLE_CMAP)
    axes[2].set_title("|Predicted - Exact|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(cs2, ax=axes[2])

    if suptitle:
        plt.suptitle(suptitle)

    finalize_plot(plt, save_path)
    return float(rel_l2)
