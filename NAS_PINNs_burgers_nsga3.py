import sys

sys.dont_write_bytecode = True

from optimizers.burgers.nsga3 import main


if __name__ == "__main__":
    main()
