from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

from one_body import one_body_correlator

def two_body_operator(Z, Spin, Isospin, momenta):
    # Spin (0, 0)
    # Isospin (1, -1)
    # momenta.shape = (2,2)

    # fourier amplitudes
    c0_plus  = Z.H.Lattice.fourier(momenta[0], +1)
    c0_minus = Z.H.Lattice.fourier(momenta[0], -1)
    c1_plus  = Z.H.Lattice.fourier(momenta[1], +1)
    c1_minus = Z.H.Lattice.fourier(momenta[1], -1)

    if Spin[0] == 1 and Isospin[0] == 1 and Spin[1] in (-1, +1) and Isospin[1] in (-1, +1):
        if Spin[1] == -1 and Isospin[1] == -1:
            operator0 = Z.H.destroy_particle
            operator1 = Z.H.destroy_particle
        elif Spin[1] == +1 and Isospin[1] == +1:
            operator0 = Z.H.create_particle
            operator1 = Z.H.create_particle
        elif Spin[1] == -1 and Isospin[1] == +1:
            operator0 = Z.H.destroy_hole
            operator1 = Z.H.destroy_hole
        elif Spin[1] == +1 and Isospin[1] == -1:
            operator0 = Z.H.create_hole
            operator1 = Z.H.create_hole
        else:
            raise ValueError(f'(S, Sz) = {Spin} and (I, Iz) = {Isospin} not allowed')

        o0_plus  = Z.H.operator(c0_plus,  operator0)
        o0_minus = Z.H.operator(c0_minus, operator0)
        o1_plus  = Z.H.operator(c1_plus,  operator1)
        o1_minus = Z.H.operator(c1_minus, operator1)

        return np.stack((
            o0_plus @ o1_plus,
            o0_plus @ o1_minus,
            o0_minus @ o1_plus,
            o0_minus @ o1_minus
            ))

    raise NotImplementedError('Currently only the dumbed spin/isospin channel is implemented.')

def two_body_correlator(Z, Spin, Isospin, total_momentum):
    lattice = Z.H.Lattice

    operators = []
    for p, k in lattice.momentum_pairs_totaling(total_momentum):
        momenta = np.stack((p, k))
        operators.append(two_body_operator(Z, Spin, Isospin, momenta))

    C = np.zeros((len(operators), len(operators), 4, 4, len(Z.taus)), dtype=complex)
    for i, sink in enumerate(operators):
        for j, source in enumerate(operators):
            C[i,j] = Z.correlator_matrix(sink, source)

    return C

if __name__ == '__main__':


    parser = beehive.parse.ArgumentParser('L', 'U', 'beta', 'nt', 'momentum', )
    parser.add_argument('--Spin', type=int, nargs=2, default=(+1, +1))
    parser.add_argument('--Isospin', type=int, nargs=2, default=(+1, +1))
    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt)

    C = two_body_correlator(Z, args.Spin, args.Isospin, args.momentum)

    for i, j in product(range(C.shape[0]), range(C.shape[1])):
        fig, ax = Z.plot_correlator(C[i,j])
        fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} S={args.Spin[0]} Sz={args.Spin[1]} I={args.Isospin[0]} Iz={args.Isospin[1]} P={args.momentum} p={i}, {j}')

    plt.show()
