import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from Pymoo.Pymoo_NSGA2 import NSGA2 as LocalNSGA2
from Pymoo.Pymoo_NSGA3 import NSGA3 as LocalNSGA3
from Poisson_2D_Search_Common import MASK_LEVELS, NUM_HIDDEN_LAYERS, SEED, proxy_evaluate


LOCAL_POP_SIZE = 24
LOCAL_N_GEN = 12
LOCAL_REF_PARTITIONS = 23


def decode_masks(x):
    return np.clip(np.round(np.asarray(x)).astype(int), 0, len(MASK_LEVELS) - 1)


class PoissonPymooProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=NUM_HIDDEN_LAYERS,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(NUM_HIDDEN_LAYERS),
            xu=np.full(NUM_HIDDEN_LAYERS, len(MASK_LEVELS) - 1),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        masks = decode_masks(x)
        loss_val, complexity = proxy_evaluate(masks)
        out["F"] = [loss_val, complexity]


def _select_best_solution(result):
    candidate_masks = np.atleast_2d(result.X)
    candidate_scores = np.atleast_2d(result.F)

    best_idx = int(np.argmin(candidate_scores[:, 0]))
    best_masks = decode_masks(candidate_masks[best_idx])

    metadata = {
        "candidate_count": int(candidate_masks.shape[0]),
        "best_proxy_objectives": [float(value) for value in candidate_scores[best_idx].tolist()],
    }
    return best_masks, metadata


def run_local_nsga2_search(*, pop_size=LOCAL_POP_SIZE, n_gen=LOCAL_N_GEN, seed=SEED):
    print("Starting local-pymoo NSGA-II architecture search...")
    problem = PoissonPymooProblem()
    algorithm = LocalNSGA2(pop_size=pop_size)

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    best_masks, metadata = _select_best_solution(result)
    metadata.update(
        {
            "search_backend": "local_pymoo",
            "pop_size": pop_size,
            "n_gen": n_gen,
        }
    )

    print("Best mask indices :", best_masks.tolist())
    print("Best neuron widths:", [MASK_LEVELS[int(mask)] for mask in best_masks])
    print("Proxy objectives  :", metadata["best_proxy_objectives"])
    return best_masks, metadata


def run_local_nsga3_search(
    *,
    n_gen=LOCAL_N_GEN,
    ref_partitions=LOCAL_REF_PARTITIONS,
    seed=SEED,
):
    print("Starting local-pymoo NSGA-III architecture search...")
    problem = PoissonPymooProblem()
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=ref_partitions)
    algorithm = LocalNSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs))

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    best_masks, metadata = _select_best_solution(result)
    metadata.update(
        {
            "search_backend": "local_pymoo",
            "pop_size": int(len(ref_dirs)),
            "n_gen": n_gen,
            "ref_partitions": ref_partitions,
        }
    )

    print("Best mask indices :", best_masks.tolist())
    print("Best neuron widths:", [MASK_LEVELS[int(mask)] for mask in best_masks])
    print("Proxy objectives  :", metadata["best_proxy_objectives"])
    return best_masks, metadata
