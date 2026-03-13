import matplotlib.pyplot as plt
import numpy as np

def plot_results(x, y, u_pred, u_true, domain, save_dir, rel_l2):
    error = np.abs(u_pred - u_true)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.title("Exact Solution")
    plt.imshow(u_true, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="YlGnBu")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("Predicted Solution")
    plt.imshow(u_pred, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="YlGnBu")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("|Pred - Exact|")
    plt.imshow(error, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="YlGnBu")
    plt.colorbar()
    plt.suptitle(f"{domain.capitalize()} Domain\nRel L² = {rel_l2:.4e}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/result_comparison.png", dpi=300)
    plt.close()
