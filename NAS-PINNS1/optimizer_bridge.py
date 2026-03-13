#!/usr/bin/env python3
"""Bridge for loading reference Pymoo optimizer files without editing them."""

from pathlib import Path
from types import ModuleType
from typing import Tuple


def _sanitize_source_for_exec(source: str) -> str:
    # Keep reference files unchanged on disk and remove doc-markers only in memory.
    return "\n".join(line for line in source.splitlines() if line.strip() != "[docs]")


def _load_reference_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing reference optimizer file: {file_path}")
    source = file_path.read_text(encoding="utf-8")
    module = ModuleType(module_name)
    module.__file__ = str(file_path)
    exec(compile(_sanitize_source_for_exec(source), str(file_path), "exec"), module.__dict__)
    return module


def load_pymoo_reference_optimizers(pymoo_dir: Path) -> Tuple[type, type, type]:
    nsga2_mod = _load_reference_module("ref_pymoo_nsga2", pymoo_dir / "Pymoo_NSGA2.py")
    nsga3_mod = _load_reference_module("ref_pymoo_nsga3", pymoo_dir / "Pymoo_NSGA3.py")
    pso_mod = _load_reference_module("ref_pymoo_pso", pymoo_dir / "Pymoo_PSO.py")
    return nsga2_mod.NSGA2, nsga3_mod.NSGA3, pso_mod.PSO
