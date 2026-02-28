import os


PAPER_BASELINE = {
    "stage": "lbfgs",
    "skip_lbfgs": False,
    "use_pso": False,
    "pso_iters": 8,
    "pso_swarm": 16,
    "pso_span": 0.25,
    "epochs": 12000,
    "layers": 5,
    "base_neurons": 128,
    "train_nt": 20,
    "train_nx": 25,
    "train_ny": 25,
    "test_nt": 41,
    "test_nx": 500,
    "test_ny": 500,
    "eval_batch_size": 65536,
    "slice_grid": 200,
    "slice_times": "0,1,2",
    "repeats": 5,
    "seed": 42,
}


OURS_FAST = {
    "stage": "pso",
    "skip_lbfgs": False,
    "use_pso": True,
    "pso_iters": 8,
    "pso_swarm": 16,
    "pso_span": 0.25,
    "epochs": 12000,
    "layers": 5,
    "base_neurons": 128,
    "train_nt": 20,
    "train_nx": 25,
    "train_ny": 25,
    "test_nt": 41,
    "test_nx": 500,
    "test_ny": 500,
    "eval_batch_size": 65536,
    "slice_grid": 200,
    "slice_times": "0,1,2",
    "repeats": 5,
    "seed": 42,
}


PROFILE_MAP = {
    "paper_baseline": PAPER_BASELINE,
    "ours_fast": OURS_FAST,
}


LOCKED_KEYS = {
    "paper_baseline": set(PAPER_BASELINE.keys()),
    "ours_fast": set(),
}


def apply_profile(args, method_name):
    if args.profile not in PROFILE_MAP:
        raise ValueError(f"Unknown profile: {args.profile}")

    profile_cfg = PROFILE_MAP[args.profile]
    locked = LOCKED_KEYS.get(args.profile, set())

    for key, value in profile_cfg.items():
        current = getattr(args, key, None)
        if key in locked and current != value:
            raise ValueError(
                f"Profile '{args.profile}' locks --{key.replace('_', '-')}: expected {value}, got {current}"
            )
        setattr(args, key, value)

    if getattr(args, "save_dir", None) in (None, ""):
        args.save_dir = os.path.join("results", "burgers2d", method_name, args.profile)

    return args


def parse_slice_times(slice_times):
    return [float(v.strip()) for v in slice_times.split(",") if v.strip()]
