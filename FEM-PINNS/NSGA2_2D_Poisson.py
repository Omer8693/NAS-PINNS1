import os
import time

import numpy as np

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
except ImportError as exc:
    raise ImportError("This script requires pymoo. Install with: pip install pymoo") from exc

from Poisson_2D_Common import device, save_json, start_run_logging
from Poisson_2D_Search_Common import (
    ADAM_EPOCHS,
    BASE_NEURONS,
    MASK_LEVELS,
    NUM_HIDDEN_LAYERS,
    PROXY_EPOCHS,
    SEED,
    architecture_summary,
    evaluate_search_model,
    print_architecture,
    proxy_evaluate,
    train_final_model,
)


POP_SIZE = 24
N_GEN = 12


class PoissonNSGA2Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=NUM_HIDDEN_LAYERS,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(NUM_HIDDEN_LAYERS),
            xu=np.full(NUM_HIDDEN_LAYERS, len(MASK_LEVELS) - 1),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        masks = np.clip(np.round(x).astype(int), 0, len(MASK_LEVELS) - 1)
        loss_val, complexity = proxy_evaluate(masks)
        out["F"] = [loss_val, complexity]


def run_nsga2_search():
    print("Starting NSGA-II architecture search...")
    problem = PoissonNSGA2Problem()
    algorithm = NSGA2(pop_size=POP_SIZE)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", N_GEN),
        seed=SEED,
        verbose=True,
    )

    candidate_masks = np.atleast_2d(res.X)
    candidate_scores = np.atleast_2d(res.F)

    best_idx = int(np.argmin(candidate_scores[:, 0]))
    best_masks = np.clip(
        np.round(candidate_masks[best_idx]).astype(int),
        0,
        len(MASK_LEVELS) - 1,
    )

    print("Best mask indices :", best_masks.tolist())
    print("Best neuron widths:", [MASK_LEVELS[int(mask)] for mask in best_masks])
    print("Proxy objectives  :", candidate_scores[best_idx].tolist())
    return best_masks


def main():
    save_dir = "./2D-Poisson-NSGA2"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)

        total_start = time.time()
        best_masks = run_nsga2_search()
        model, loss_history, adam_time, lbfgs_time = train_final_model(best_masks)

        print_architecture(model, best_masks, "DISCRETE NSGA-II / NAS-PINN ARCHITECTURE")

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
            "pop_size": POP_SIZE,
            "n_gen": N_GEN,
            "loss_history": loss_history,
        }

        save_json(os.path.join(save_dir, "NSGA2_results.json"), results)
        save_json(os.path.join(save_dir, "NSGA2_evaluation.json"), evaluation, indent=4)

        print(f"Adam time   : {adam_time:.4f} s")
        print(f"L-BFGS time : {lbfgs_time:.4f} s")
        print(f"Eval time   : {eval_time:.4f} s")
        print(f"Total time  : {total_time:.4f} s")
        print(f"Relative L2 : {rel_l2:.8e}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
