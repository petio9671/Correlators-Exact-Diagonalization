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

def mega(C):
    reordered = C.transpose((0, 2, 1, 3, 4))
    return reordered.reshape((reordered.shape[0]*reordered.shape[1], reordered.shape[2]*reordered.shape[3], reordered.shape[4]))

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
            C = tuple(mega(two_body_correlator(z, Spin, Isospin, TotalMomentum)) for z in Z)
            # and make a megafigure.
            sinks, sources = C[0].shape[:2]
            fig, ax = plt.subplots(sources, sinks, figsize=(3*sinks, 2*sources), sharex='col')
            for i, j in product(range(sinks), range(sources)):
                for z, c in zip(Z, C):
                    ax[i,j].plot(z.taus, c[i, j].real, label=f'real nt={z.nt}', **style(z.nt))
                    ax[i,j].plot(z.taus, c[i, j].imag, label=f'imag nt={z.nt}', **style(z.nt))

            ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fig.suptitle(f'{lattice} U={hubbard.U} β={args.beta} (S={Spin[0]} Sz={Spin[1]}) (I={Isospin[0]} Iz={Isospin[1]}) P={TotalMomentum}')
            fig.tight_layout()
            pdf.save(fig)
