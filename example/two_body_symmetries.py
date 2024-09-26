from itertools import product
import numpy as np
import scipy as sc

import beehive
beehive.format='csc'
beehive.sparse_array=sc.sparse.csc_array

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from one_body import one_body_correlator
from two_body import two_body_correlator
from pdf import PDF

def transform(C, S, Sz, I, Iz):
    
    if Iz == -1:
        # These are NOT the full suite of symmetries we might know about!
        # For example, we don't handle the momentum permutation symmetries.
        # Sinilkov table 4.3 I  1234    Q=-2 Sz=+1
        # Sinilkov table 4.3 XF 1234    Q=+2 Sz=+1
        return C[...,::-1,::-1,:]

    return C

def mega(C):
    reordered = C.transpose((0, 2, 1, 3, 4))
    return reordered.reshape((reordered.shape[0]*reordered.shape[1], reordered.shape[2]*reordered.shape[3], reordered.shape[4]))


if __name__ == '__main__':


    parser = beehive.cli.ArgumentParser(('L', 'U', 'beta', 'nt', ))
    parser.add_argument('pdf', type=str, default='')

    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt)
    logger.info(lattice)
    logger.info(hubbard)
    logger.info(Z)

    logger.info('CALCULATING')
    channel = dict()
    for Spin, Isospin, TotalMomentum in product(
            ((0, 0), (1, +1), (1, 0), (1, -1)),
            ((0, 0), (1, +1), (1, 0), (1, -1)),
            lattice.momenta,
            ):
        (S, Sz) = Spin
        (I, Iz) = Isospin
        (px, py) = TotalMomentum
        logger.info(f'{S=} {Sz=} {I=} {Iz=} P=({px}, {py})')
        C = two_body_correlator(Z, Spin, Isospin, TotalMomentum)
        channel[(Spin, Isospin, (TotalMomentum[0], TotalMomentum[1]))] = C


    logger.info('PLOTTING')
    with PDF(args.pdf) as pdf:
        for (S, Sz), (I, Iz), (px, py) in product(
                ((0, (0,)), (1, (+1, 0, -1))),
                ((0, (0,)), (1, (+1, -1)), (1, (0,))),
                lattice.momenta,
                ):

            logger.info(f'{S=} {Sz=} {I=} {Iz=} P=({px}, {py})')
            M = mega(channel[((S, Sz[0]), (I, Iz[0]), (px, py))])
            fig, ax = plt.subplots(M.shape[0], M.shape[1], figsize=np.array([M.shape[0], M.shape[1]]) * np.array([3, 2]), sharex='col')

            for n, (sz, iz) in enumerate(product(Sz, Iz)):
                C = mega(transform(channel[((S, sz), (I, iz), (px, py))], S, sz, I, iz))
                offset = n * Z.taus[1] / 10
                for i, j in product(range(M.shape[0]), range(M.shape[1])):
                    ax[i,j].plot(Z.taus[1:]+offset, C[i,j,1:].real, label=f'Re(Sz={sz} Iz={iz})')
                    ax[i,j].plot(Z.taus[1:]+offset, C[i,j,1:].imag, label=f'Im(Sz={sz} Iz={iz})')

            for j in range(M.shape[1]):
                ax[-1, j].set_xlim([0, Z.beta])

            ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} S={S} Sz={Sz} I={I} Iz={Iz} P=({px}, {py})', fontsize=18)

            pdf.save(fig)
