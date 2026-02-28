import os


PAPER_BASELINE = {
    "stage": "lbfgs",
    "skip_lbfgs": False,
    "use_pso": False,
    "pso_iters": 8,
    "pso_swarm": 16,
    "pso_span": 0.25,
    "epochs": 12000,
    "layers": 4,
    "base_neurons": 128,
    "train_nt": 40,
    "train_nx": 120,
    "test_nt": 40,
    "test_nx": 120,
    "beta_list": "1.0,0.5,0.1",
    "paper_betas": "1.0,0.5,0.1",
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
    "layers": 4,
    "base_neurons": 128,
    "train_nt": 40,
    "train_nx": 120,
    "test_nt": 40,
    "test_nx": 120,
    "beta_list": "1.0,0.5,0.1",
    "paper_betas": "1.0,0.5,0.1",
    "repeats": 5,
    "seed": 42,
}


PROFILE_MAP = {
    "paper_baseline": PAPER_BASELINE,
    "ours_fast": OURS_FAST,
}


LOCKED_KEYS = {
    "paper_baseline": set(PAPER_BASELINE.keys()) - {"beta"},
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
        args.save_dir = os.path.join("results", "advection", method_name, args.profile)

    return args


def parse_beta_list(beta_list):
    return [float(v.strip()) for v in beta_list.split(",") if v.strip()]
