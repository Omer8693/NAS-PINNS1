import sys

sys.dont_write_bytecode = True

from optimizers.advection.bayesian import main


if __name__ == "__main__":
    main()
