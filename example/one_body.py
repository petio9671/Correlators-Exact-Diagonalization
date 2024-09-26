from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)
from beehive.monitoring import Timed

_one_body_operator_cache = dict()
def indexed_operator(H, amplitudes, idx):
    # With a fixed setup H and given set of amplitudes we can
    # still construct four independent operators, p, p†, h and h†.
    try:
        return _one_body_operator_cache[(H, tuple(amplitudes.tolist()), idx)]
    except:
        pass
    if idx== 0:
        o = H.operator(amplitudes, H.destroy_particle)
    if idx== 1:
        o = H.operator(amplitudes, H.create_particle)
    if idx== 2:
        o = H.operator(amplitudes, H.destroy_hole)
    if idx== 3:
        o = H.operator(amplitudes, H.create_hole)
    _one_body_operator_cache[(H, tuple(amplitudes.tolist()), idx)] = o
    return o

@Timed(logging.info)
def one_body_operators(Z, momentum, operator):
    """Calculate a one-body operator in the momentum space and 2 bands

    Args:
        Z (obj): Partition function object
        momentum (ndarray): Momentum that we want the operators to be projected on
        operator (list of sparse_array): Operator in position space

    Returns:
        list of sparse_array: One-body operator in momentum space with 2 bands
        
    """

    # fourier amplitudes
    c_plus  = Z.H.Lattice.fourier(momentum, +1)
    c_minus = Z.H.Lattice.fourier(momentum, -1)

    # band operators
    o_plus  = Z.H.operator(c_plus,  operator)
    o_minus = Z.H.operator(c_minus, operator)

    return list((o_plus, o_minus))

@Timed(logging.info)
def one_body_correlator(Z, hubbard_species, momentum):
    """Calculate a one-body correlator matrix for given species in the momentum space 

    Args:
        Z (obj): partition function object
        hubbard_species (str): Operator species (particle or a hole)
        momentum (ndarray): Momentum that we want the operators to be projected on
        

    Returns:
        ndarray: Correlator matrix
        
    """

    logger.debug(f'{hubbard_species} with {momentum=}')
    flavor = Z.H.destroy_particle if (hubbard_species == 'particle') else Z.H.destroy_hole
    operators = one_body_operators(Z, momentum, flavor)

    return Z.correlator_matrix(operators, operators)

if __name__ == '__main__':

    parser = beehive.cli.ArgumentParser(('L', 'U', 'beta', 'nt', 'momentum', 'species'))
    parser.add_argument('--pdf', type=str, default='')
    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L) # Instantiate the lattice
    hubbard = beehive.Hubbard(lattice, args.U) # Instantiate the Hubbard model

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt) # Instantiate the partition function
    logger.info(Z)

    momentum = lattice.momenta[args.momentum] # Get the momentum you want from the momenta array
    C = one_body_correlator(Z, args.species, momentum) # Calculate the one-body correlation function

    # Plot the correlation function
    fig,ax = Z.plot_correlator(C)

    # We expect to have 0 at the non-diagonal elements
    ax[0,0].set_yscale('log')
    ax[1,1].set_yscale('log')

    fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} {args.species} p={momentum}')

    if args.pdf:
        fig.savefig(args.pdf)
    else:
        plt.show()
