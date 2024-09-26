import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

if __name__ == '__main__':


    import argparse

    parser = beehive.cli.ArgumentParser(('L'))
    parser.add_argument('--bz', default=False, action='store_true', help='Show the Brillouin zone rather than the spatial lattice')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L)
    print(lattice)

    if args.bz:
        print(lattice.momenta)

    fig, ax = plt.subplots(1,1)
    
    if args.bz:
        lattice.plot_bz(ax)
    else:
        lattice.plot_lattice(ax)
        ax.set_aspect('auto')

    fig.suptitle(f'{lattice} {"Brillouin Zone" if args.bz else "Layout"}')

    if args.pdf:
        fig.savefig(args.pdf)
    else:
        plt.show()
