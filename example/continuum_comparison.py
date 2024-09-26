from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

from one_body import one_body_correlator
from two_body import two_body_correlator
from pdf import PDF

if __name__ == '__main__':

    parser = beehive.cli.ArgumentParser(('L', 'U', 'beta'))
    parser.add_argument('--nts', type=beehive.cli.nt, nargs="*")
    parser.add_argument('pdf', type=str, default='')
    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)

    Z = [ beehive.PartitionFunction(hubbard, args.beta, nt) for nt in args.nts]

    for z in Z:
        logger.info(z)

    def style(nt):
        if nt < float('inf'):
            return {
                    'linestyle': 'none',
                    'marker': '.',
                    }
        else:
            return {
                    'marker': 'none'
                    }

    with PDF(args.pdf) as pdf:
        for momentum, species in product(lattice.momenta, ('particle', 'hole')):
            # For each species + (conserved) momentum build a 2x2 band-space figure
            fig, ax = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
            # which shows the correlator for all the nts requested
            for z in Z:
                C = one_body_correlator(z, species, momentum)
                for i,j in product(range(2), range(2)):
                    ax[i,j].plot(z.taus, C[i,j].real, label=f'real nt={z.nt}', **style(z.nt))
                    ax[i,j].plot(z.taus, C[i,j].imag, label=f'imag nt={z.nt}', **style(z.nt))

            ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # Since momentum is conserved the diagonals are always <o†o> >= 0.
            ax[0,0].set_yscale('log')
            ax[1,1].set_yscale('log')
            fig.suptitle(f'{lattice} U={hubbard.U} β={args.beta} {species} p={momentum}')
            fig.tight_layout()
            pdf.save(fig)

        for Spin, Isospin, TotalMomentum in product(
                ((0, 0), (1, +1), (1, 0), (1, -1)),
                ((0, 0), (1, +1), (1, 0), (1, -1)),
                lattice.momenta,
                ):
            # For each channel, compute the correlators
            C = tuple(two_body_correlator(z, Spin, Isospin, TotalMomentum) for z in Z)
            # and make a figure for every shell/shell combination.
            for i, j in product(range(C[0].shape[0]), range(C[0].shape[1])):
                fig, ax = plt.subplots(4, 4, figsize=(12, 8), sharex='col')
                for z, c in zip(Z, C):
                    for row, col in product(range(4), range(4)):
                        ax[row, col].plot(z.taus, c[i, j, row, col].real, label=f'real nt={z.nt}', **style(z.nt))
                        ax[row, col].plot(z.taus, c[i, j, row, col].imag, label=f'imag nt={z.nt}', **style(z.nt))

                ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                fig.suptitle(f'{lattice} U={hubbard.U} β={args.beta} (S={Spin[0]} Sz={Spin[1]}) (I={Isospin[0]} Iz={Isospin[1]}) P={TotalMomentum} p={i}, {j}')
                fig.tight_layout()
                pdf.save(fig)
