from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

def one_body_operators(Z, momentum, operator):

    # fourier amplitudes
    c_plus  = Z.H.Lattice.fourier(momentum, +1)
    c_minus = Z.H.Lattice.fourier(momentum, -1)

    # band operators
    o_plus  = Z.H.operator(c_plus,  operator)
    o_minus = Z.H.operator(c_minus, operator)

    return np.stack((o_plus, o_minus))

def one_body_correlator(Z, hubbard_species, momentum):
    flavor = Z.H.destroy_particle if (hubbard_species == 'particle') else Z.H.destroy_hole
    operators = one_body_operators(Z, momentum, flavor)
    return Z.correlator_matrix(operators, operators)

if __name__ == '__main__':


    parser = beehive.parse.ArgumentParser('L', 'U', 'beta', 'nt', 'momentum', 'species')
    parser.add_argument('--versions', default=False, action='store_true')

    args = parser.parse_args()

    if args.versions:
        print('numpy', np.__version__)
        print('scipy', sc.__version__)
        exit(0)

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt)

    momentum = lattice.momenta[args.momentum]
    C = one_body_correlator(Z, args.species, momentum)

    fig,ax = Z.plot_correlator(C)
    ax[0,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} {args.species} p={momentum}')

    plt.show()
