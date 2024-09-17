import numpy as np
import exact
import matplotlib.pyplot as plt

import scipy as sc

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, nargs=2, default=(3,3))
    parser.add_argument('--bz', default=False, action='store_true')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()

    lattice = exact.Honeycomb(*args.L)

    print(lattice.momenta)

    fig, ax = plt.subplots(1,1)
    
    if args.bz:
        lattice.plot_bz(ax)
    else:
        lattice.plot_lattice(ax)
        ax.set_aspect('auto')

    if args.pdf:
        fig.savefig(args.pdf)
    else:
        plt.show()
