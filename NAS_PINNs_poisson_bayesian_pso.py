import sys

sys.dont_write_bytecode = True

from optimizers.pso.runner import main


if __name__ == "__main__":
    sys.argv.extend(["--target", "poisson-bayesian"])
    main()
