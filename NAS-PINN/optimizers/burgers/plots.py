import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat


interactive_plots = False
SIMPLE_CMAP = "YlGnBu"
EXACT_NU = 0.01
NU_TOL = 1e-12


def should_use_exact_plots(nu_value):
    return abs(float(nu_value) - EXACT_NU) <= NU_TOL


def finalize_plot(save_path, use_interactive=interactive_plots):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if use_interactive:
        plt.show()
    else:
        plt.close()


def plot_burgers_exact_pred_error(
    x_vals,
    t_vals,
    exact_u,
    pred_u,
    save_path,
    pred_title="Predicted (NAS-PINN)",
    use_interactive=interactive_plots,
):
    Xg, Tg = np.meshgrid(x_vals, t_vals, indexing="ij")
    abs_err = np.abs(pred_u - exact_u)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cs0 = axes[0].contourf(Xg, Tg, exact_u, levels=60, cmap=SIMPLE_CMAP)
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(Xg, Tg, pred_u, levels=60, cmap=SIMPLE_CMAP)
    axes[1].set_title(pred_title)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(Xg, Tg, abs_err, levels=60, cmap=SIMPLE_CMAP)
    axes[2].set_title("|Pred - Exact|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(cs2, ax=axes[2])

    finalize_plot(save_path, use_interactive=use_interactive)


def plot_burgers_heatmap(
    model,
    device,
    x_min,
    x_max,
    t_min,
    t_max,
    save_path,
    title="Burgers solution learned with NAS-PINN (heatmap)",
    use_interactive=interactive_plots,
):
    nx, nt = 200, 100
    x_grid = torch.linspace(x_min, x_max, nx, device=device).unsqueeze(1)
    t_grid = torch.linspace(t_min, t_max, nt, device=device).unsqueeze(1)
    X, T = torch.meshgrid(x_grid.squeeze(), t_grid.squeeze(), indexing="ij")
    XT = torch.cat([X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)], dim=1)

    with torch.no_grad():
        U = model(XT).cpu().reshape(nx, nt)

    plt.figure(figsize=(10, 6))
    cs = plt.contourf(X.cpu().numpy(), T.cpu().numpy(), U.numpy(), levels=60, cmap=SIMPLE_CMAP)
    plt.colorbar(cs, label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()
    finalize_plot(save_path, use_interactive=use_interactive)


def plot_burgers_time_slices(
    model,
    device,
    x_min,
    x_max,
    save_path,
    t_values=None,
    nx=400,
    title="Time slices of the learned Burgers solution",
    use_interactive=interactive_plots,
):
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    x_line = torch.linspace(x_min, x_max, nx, device=device).unsqueeze(1)

    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        for t_val in t_values:
            t_line = torch.full_like(x_line, float(t_val))
            u_line = model(torch.cat([x_line, t_line], dim=1)).cpu().numpy().squeeze()
            plt.plot(x_line.cpu().numpy().squeeze(), u_line, label=f"t={t_val:.2f}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(save_path, use_interactive=use_interactive)


def plot_burgers_time_slices_with_exact(
    model,
    device,
    save_path,
    mat_path="burgers_shock.mat",
    t_values=None,
    use_interactive=interactive_plots,
):
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    try:
        data = loadmat(mat_path)
    except Exception as exc:
        print(f"Could not load exact solution file '{mat_path}': {exc}")
        return

    x_exact = data["x"].squeeze()
    t_exact = data["t"].squeeze()
    u_exact = np.real(data["usol"])
    x_tensor = torch.tensor(x_exact, dtype=torch.float32, device=device).unsqueeze(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(t_values)))

    with torch.no_grad():
        for idx, t_val in enumerate(t_values):
            t_idx = int(np.argmin(np.abs(t_exact - t_val)))
            t_match = float(t_exact[t_idx])
            t_tensor = torch.full_like(x_tensor, t_match)
            u_pred = model(torch.cat([x_tensor, t_tensor], dim=1)).cpu().numpy().squeeze()
            u_ref = u_exact[:, t_idx].squeeze()

            axes[0].plot(x_exact, u_ref, "-", linewidth=2.5, color=colors[idx], label=f"t={t_match:.2f}")
            axes[1].plot(x_exact, u_pred, "-", linewidth=2.5, color=colors[idx], label=f"t={t_match:.2f}")

    axes[0].set_title("Exact time slices (Burgers)")
    axes[1].set_title("Predicted time slices (Burgers)")
    axes[0].set_xlabel("x")
    axes[1].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(title="Exact", ncol=1)
    axes[1].legend(title="Pred", ncol=1)

    fig.suptitle("Exact vs Predicted time slices (Burgers)")
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    finalize_plot(save_path, use_interactive=use_interactive)


def plot_burgers_full_exact_vs_pred(
    model,
    device,
    save_path,
    mat_path="burgers_shock.mat",
    pred_title="Predicted (NAS-PINN)",
    use_interactive=interactive_plots,
):
    try:
        data = loadmat(mat_path)
    except Exception as exc:
        print(f"Could not load exact solution file '{mat_path}': {exc}")
        return None

    x_exact = data["x"].squeeze()
    t_exact = data["t"].squeeze()
    u_exact = np.real(data["usol"])

    Xg, Tg = np.meshgrid(x_exact, t_exact, indexing="ij")
    xt_np = np.stack([Xg.ravel(), Tg.ravel()], axis=1)
    xt = torch.tensor(xt_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xt).cpu().numpy().reshape(len(x_exact), len(t_exact))

    rel_l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)
    print(f"Relative L2 error (full grid): {rel_l2:.4e}")

    plot_burgers_exact_pred_error(
        x_exact,
        t_exact,
        u_exact,
        u_pred,
        save_path,
        pred_title=pred_title,
        use_interactive=use_interactive,
    )
    return float(rel_l2)


def plot_loss_curve(loss_values, save_path, title="Training Loss", yscale="log", use_interactive=interactive_plots):
    if not loss_values:
        print("Skipping loss plot: empty history.")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(loss_values, linewidth=1.2, color="royalblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if yscale:
        plt.yscale(yscale)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(save_path, use_interactive=use_interactive)
