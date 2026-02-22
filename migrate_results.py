import argparse
import shutil
from pathlib import Path


RESULT_DIR_MAP = {
    "results_plots1": "results/burgers/naspinn",
    "results_plots_nsga2": "results/burgers/nsga2",
    "results_plots_nsga3": "results/burgers/nsga3",
    "results_plots": "results/poisson/naspinn",
    "results_plots_poisson_bayes": "results/poisson/bayesian",
    "results/burgers/bilevel": "results/burgers/naspinn",
    "results/poisson/bilevel": "results/poisson/naspinn",
}

CHECKPOINT_FILE_MAP = {
    "checkpoint_last.pth": "results/burgers/naspinn/checkpoint_last.pth",
    "checkpoint_last_nsga2.pth": "results/burgers/nsga2/checkpoint_last_nsga2.pth",
    "checkpoint_last_nsga3.pth": "results/burgers/nsga3/checkpoint_last_nsga3.pth",
    "poisson_checkpoint_last.pth": "results/poisson/naspinn/poisson_checkpoint_last.pth",
}

EXPECTED_RESULT_DIRS = [
    "results/burgers/naspinn",
    "results/burgers/nsga2",
    "results/burgers/nsga3",
    "results/burgers/bayesian",
    "results/poisson/naspinn",
    "results/poisson/nsga2",
    "results/poisson/nsga3",
    "results/poisson/bayesian",
]


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path, dry_run: bool):
    if not src.exists():
        return

    if dry_run:
        print(f"[DRY] move: {src} -> {dst}")
        return

    ensure_parent(dst)
    if src.is_dir() and dst.exists() and dst.is_dir():
        for child in src.iterdir():
            move_path(child, dst / child.name, dry_run=False)
        src.rmdir()
    else:
        shutil.move(str(src), str(dst))
        print(f"moved: {src} -> {dst}")


def ensure_expected_dirs(root: Path, dry_run: bool):
    for rel in EXPECTED_RESULT_DIRS:
        path = root / rel
        if dry_run:
            if not path.exists():
                print(f"[DRY] mkdir: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)


def fix_misplaced_files(root: Path, dry_run: bool):
    burgers_bilevel = root / "results" / "burgers" / "naspinn"
    burgers_bayesian = root / "results" / "burgers" / "bayesian"
    poisson_bilevel = root / "results" / "poisson" / "naspinn"

    if burgers_bilevel.exists():
        for file_path in burgers_bilevel.iterdir():
            if file_path.is_file() and file_path.name.startswith("bayes"):
                move_path(file_path, burgers_bayesian / file_path.name, dry_run=dry_run)

    if poisson_bilevel.exists():
        for file_path in poisson_bilevel.iterdir():
            if file_path.is_file() and file_path.name.startswith("burgers_"):
                move_path(file_path, burgers_bilevel / file_path.name, dry_run=dry_run)


def migrate(root: Path, dry_run: bool):
    ensure_expected_dirs(root, dry_run=dry_run)

    for old_rel, new_rel in RESULT_DIR_MAP.items():
        src = root / old_rel
        dst = root / new_rel
        move_path(src, dst, dry_run=dry_run)

    for old_rel, new_rel in CHECKPOINT_FILE_MAP.items():
        src = root / old_rel
        dst = root / new_rel
        move_path(src, dst, dry_run=dry_run)

    fix_misplaced_files(root, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate legacy results_plots folders to new results/* structure")
    parser.add_argument("--root", type=str, default=".", help="project root path")
    parser.add_argument("--dry-run", action="store_true", help="print planned operations without moving files")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    migrate(root, args.dry_run)
    print("Migration completed.")


if __name__ == "__main__":
    main()
