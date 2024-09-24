import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':


    import argparse

    parser = beehive.parse.ArgumentParser('L', 'U')
    parser.add_argument('--pdf', default='', type=str)
    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)
    print(hubbard)

    energies, unitary = np.linalg.eigh(hubbard.Hamiltonian.toarray())

    fig, ax = plt.subplots(1,1)
    ax.plot(energies, marker='o', linestyle='none')

    ax.set_xlabel('index')
    ax.set_ylabel('energy/κ')
    fig.suptitle(f'Spectrum {lattice} U={hubbard.U}')

    if args.pdf:
        fig.savefig(args.pdf)
    else:
        plt.show()
