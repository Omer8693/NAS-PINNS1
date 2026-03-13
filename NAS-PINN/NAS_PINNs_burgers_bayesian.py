import sys

sys.dont_write_bytecode = True

from optimizers.burgers.bayesian import main


if __name__ == "__main__":
    main()
