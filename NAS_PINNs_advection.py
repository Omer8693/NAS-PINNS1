import sys

sys.dont_write_bytecode = True

from optimizers.advection.naspinn import main


if __name__ == "__main__":
    main()
