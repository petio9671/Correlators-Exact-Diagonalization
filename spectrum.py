import numpy as np
import exact
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, nargs=2, default=(1,1))
    parser.add_argument('--U', type=float, default=1.0)
    parser.add_argument('--pdf', default='', type=str)
    args = parser.parse_args()

    lattice = exact.Honeycomb(*args.L)
    hubbard = exact.Hubbard(lattice, args.U)
    energies, unitary = np.linalg.eigh(hubbard.Hamiltonian.toarray())

    fig, ax = plt.subplots(1,1)
    ax.plot(energies)

    ax.set_xlabel('index')
    ax.set_ylabel('energy/κ')
    fig.suptitle(f'Spectrum {lattice} U={hubbard.U}')

    if args.pdf:
        fig.savefig(args.pdf)
    else:
        plt.show()
