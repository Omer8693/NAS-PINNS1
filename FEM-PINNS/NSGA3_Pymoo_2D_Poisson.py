import os
import time

from Poisson_2D_Common import device, save_json, start_run_logging
from Poisson_2D_Pymoo_Local_Common import LOCAL_N_GEN, LOCAL_REF_PARTITIONS, run_local_nsga3_search
from Poisson_2D_Search_Common import (
    ADAM_EPOCHS,
    BASE_NEURONS,
    MASK_LEVELS,
    NUM_HIDDEN_LAYERS,
    PROXY_EPOCHS,
    architecture_summary,
    evaluate_search_model,
    print_architecture,
    train_final_model,
)


def main():
    save_dir = "./2D-Poisson-NSGA3-Pymoo"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)

        total_start = time.time()
        best_masks, search_meta = run_local_nsga3_search()
        model, loss_history, adam_time, lbfgs_time = train_final_model(best_masks)

        print_architecture(model, best_masks, "LOCAL PYMOO NSGA-III / NAS-PINN ARCHITECTURE")

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
            "loss_history": loss_history,
            "search_backend": search_meta["search_backend"],
            "candidate_count": search_meta["candidate_count"],
            "best_proxy_objectives": search_meta["best_proxy_objectives"],
            "pop_size": search_meta["pop_size"],
            "n_gen": LOCAL_N_GEN,
            "ref_partitions": LOCAL_REF_PARTITIONS,
        }

        save_json(os.path.join(save_dir, "NSGA3_Pymoo_results.json"), results)
        save_json(os.path.join(save_dir, "NSGA3_Pymoo_evaluation.json"), evaluation, indent=4)

        print(f"Adam time   : {adam_time:.4f} s")
        print(f"L-BFGS time : {lbfgs_time:.4f} s")
        print(f"Eval time   : {eval_time:.4f} s")
        print(f"Total time  : {total_time:.4f} s")
        print(f"Relative L2 : {rel_l2:.8e}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
