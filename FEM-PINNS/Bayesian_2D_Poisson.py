import os
import time

import numpy as np

try:
    from bayes_opt import BayesianOptimization
except ImportError as exc:
    raise ImportError(
        "This script requires bayesian-optimization. Install with: pip install bayesian-optimization"
    ) from exc

from Poisson_2D_Common import device, save_json, start_run_logging
from Poisson_2D_Search_Common import (
    ADAM_EPOCHS,
    BASE_NEURONS,
    MASK_LEVELS,
    NUM_HIDDEN_LAYERS,
    PROXY_EPOCHS,
    architecture_summary,
    evaluate_search_model,
    print_architecture,
    proxy_evaluate,
    train_final_model,
)


BO_INIT_POINTS = 4
BO_ITERS = 12


def decode_masks(params_dict):
    masks = []
    for idx in range(NUM_HIDDEN_LAYERS):
        raw = params_dict[f"m{idx}"]
        masks.append(int(np.clip(np.round(raw), 0, len(MASK_LEVELS) - 1)))
    return np.array(masks, dtype=int)


def run_bayesian_search():
    print("Starting Bayesian architecture search...")

    def objective(**kwargs):
        masks = decode_masks(kwargs)
        proxy_loss, _ = proxy_evaluate(masks)
        print(f"BO eval masks={masks.tolist()} proxy_loss={proxy_loss:.4e}")
        return -proxy_loss

    pbounds = {f"m{idx}": (0, len(MASK_LEVELS) - 1) for idx in range(NUM_HIDDEN_LAYERS)}
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=17,
        verbose=2,
    )
    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_ITERS)

    best_masks = decode_masks(optimizer.max["params"])
    print("Best mask indices :", best_masks.tolist())
    print("Best neuron widths:", [MASK_LEVELS[int(mask)] for mask in best_masks])
    print("Best proxy loss   :", -float(optimizer.max["target"]))
    return best_masks


def main():
    save_dir = "./2D-Poisson-Bayesian"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)

        total_start = time.time()
        best_masks = run_bayesian_search()
        model, loss_history, adam_time, lbfgs_time = train_final_model(best_masks)

        print_architecture(model, best_masks, "DISCRETE BAYESIAN / NAS-PINN ARCHITECTURE")

        domain_pts, y_pred, y_true, rel_l2, eval_time = evaluate_search_model(model, best_masks)
        total_time = time.time() - total_start

        results = {
            "domain_pts": domain_pts,
            "y_results": y_pred,
            "y_gt": y_true,
        }

        evaluation = {
            "best_masks": [int(mask) for mask in best_masks.tolist()],
            "best_widths": [int(MASK_LEVELS[int(mask)]) for mask in best_masks.tolist()],
            "times_adam": adam_time,
            "times_lbfgs": lbfgs_time,
            "times_total": total_time,
            "times_eval": eval_time,
            "l2_rel": rel_l2,
            "arch": architecture_summary(model, best_masks),
            "mask_levels": MASK_LEVELS,
            "base_neurons": BASE_NEURONS,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "proxy_epochs": PROXY_EPOCHS,
            "adam_epochs": ADAM_EPOCHS,
            "bo_init_points": BO_INIT_POINTS,
            "bo_iters": BO_ITERS,
            "loss_history": loss_history,
        }

        save_json(os.path.join(save_dir, "Bayesian_results.json"), results)
        save_json(os.path.join(save_dir, "Bayesian_evaluation.json"), evaluation, indent=4)

        print(f"Adam time   : {adam_time:.4f} s")
        print(f"L-BFGS time : {lbfgs_time:.4f} s")
        print(f"Eval time   : {eval_time:.4f} s")
        print(f"Total time  : {total_time:.4f} s")
        print(f"Relative L2 : {rel_l2:.8e}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
