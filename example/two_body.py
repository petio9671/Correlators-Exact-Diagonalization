from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

two_body_amplitudes = {
    # For two momenta (or unit cells) k, q we can construct 16 different two-body operators,
    #
    #   {p(k), p†(k), h(k), h†(k)} {p(q), p†(q), h(q), h†(q)}
    #
    # 16 = (4 choices for the left operator) * (4 choices for the right operator).
    #
    # Except for the S=1 Sz=±1 I=1 Iz=±1 these operators don't have definite isospin.
    # Operators with good isospin can be written
    #
    #   O = ∑_{l,r} c[l,r]  {p, p†, h, h†}[l] {p, p†, h, h†}[r]
    #
    # with l, r from 0 to 3 indicating left and right operators and
    # the amplitudes c can be thought of like Clebsch-Gordan coefficients
    # that differ for every channel given by (S, Sz) and (I, Iz).
    # We encode these amplitudes in a 4x4 matrix (l and r indices) for each channel.
    #
    # The dictionary's keys are
    #
    #   ((S, Sz), (I, Iz))
    #
    # and the values are 4x4 matrices.

    # S=1 I=1 3x3 matrix
    ((1, +1), (1, +1)): np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex),
    ((1,  0), (1, +1)): np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    ((1, -1), (1, +1)): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=complex),
    ((1, +1), (1,  0)): np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=complex) / np.sqrt(2),
    ((1,  0), (1,  0)): np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0,-1], [0, 0,-1, 0]], dtype=complex) / 2,
    ((1, -1), (1,  0)): np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    ((1, +1), (1, -1)): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex),
    ((1,  0), (1, -1)): np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    ((1, -1), (1, -1)): np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex),
    # S=1 I=0 3x1
    ((1, +1), (0, 0)): np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0,-1, 0, 0]], dtype=complex) / np.sqrt(2),
    ((1,  0), (0, 0)): np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0,+1], [0, 0,-1, 0]], dtype=complex) / 2,
    ((1, -1), (0, 0)): np.array([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    # S=0 I=1 1x3
    ((0, 0), (1, +1)): np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0,-1, 0, 0], [0, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    ((0, 0), (1,  0)): np.array([[0,-1, 0, 0], [1, 0, 0, 0], [0, 0, 0,+1], [0, 0,-1, 0]], dtype=complex) / 2,
    ((0, 0), (1, -1)): np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 0]], dtype=complex) / np.sqrt(2),
    # S=0 I=0 1x1
    ((0, 0), (0,  0)): np.array([[0,+1, 0, 0], [+1, 0, 0, 0], [0, 0, 0,+1], [0, 0,+1, 0]], dtype=complex) / 2,
    }


def two_body_operator(Z, Spin, Isospin, momenta):
    """Calculate a two-body operator in the momentum space for a particular spin-isospin channel

    Args:
        Z (obj): partition function object
        Spin (tuple of int): Spin and 3rd component of the spin
        Isospin (tuple of int): Isopin and 3rd component of the isospin
        momenta (ndarray): Momenta of the two single body particles

    Returns:
        sparse_array: Two-body operator in momentum space at a particular channel with 2 band combinations 
        
    """

    # Spin (0, 0)
    # Isospin (1, -1)
    # momenta.shape = (2,2)

    try:
        a = two_body_amplitudes[(Spin, Isospin)]
    except:
        raise ValueError(f'(S, Sz) = {Spin} and (I, Iz) = {Isospin} not allowed')

    # fourier amplitudes
    c0_plus  = Z.H.Lattice.fourier(momenta[0], +1)
    c0_minus = Z.H.Lattice.fourier(momenta[0], -1)
    c1_plus  = Z.H.Lattice.fourier(momenta[1], +1)
    c1_minus = Z.H.Lattice.fourier(momenta[1], -1)

    def op(c, idx):
        if idx== 0:
            return Z.H.operator(c, Z.H.destroy_particle)
        if idx== 1:
            return Z.H.operator(c, Z.H.create_particle)
        if idx== 2:
            return Z.H.operator(c, Z.H.destroy_hole)
        if idx== 3:
            return Z.H.operator(c, Z.H.create_hole)

    stack = list(beehive.sparse_array((Z.H.Hilbert_space_dimension, Z.H.Hilbert_space_dimension)) for i in range(4))
    for (l, r) in product(range(4), range(4)):
        if a[l, r] == 0:
            #print(f'(S, Sz)={Spin} (I, Iz)={Isospin} element [{l},{r}] vanishes.')
            continue
        l_plus = op(c0_plus, l)
        l_minus= op(c0_minus, l)
        r_plus = op(c1_plus, r)
        r_minus= op(c1_minus, r)
        stack[0] += a[l,r] * l_plus @ r_plus
        stack[1] += a[l,r] * l_plus @ r_minus
        stack[2] += a[l,r] * l_minus @ r_plus
        stack[3] += a[l,r] * l_minus @ r_minus

    return stack

def two_body_correlator(Z, Spin, Isospin, total_momentum):
    """Calculate a two-body correlator with a total momentum for a particular spin-isospin channel

    Args:
        Z (obj): Partition function object
        Spin (tuple of int): Spin and 3rd component of the spin
        Isospin (tuple of int): Isopin and 3rd component of the isospin
        total_momentum (ndarray): Total momentum of the system

    Returns:
        sparse_array: Correlator matrix
        
    """

    lattice = Z.H.Lattice

    operators = []
    # Go through every possible combination that totals the momentum and save the operator
    for p, k in lattice.momentum_pairs_totaling(total_momentum):
        momenta = np.stack((p, k))
        operators.append(two_body_operator(Z, Spin, Isospin, momenta))

    C = np.zeros((len(operators), len(operators), 4, 4, len(Z.taus)), dtype=complex)
    for i, sink in enumerate(operators):
        for j, source in enumerate(operators):
            C[i,j] = Z.correlator_matrix(sink, source)

    return C

if __name__ == '__main__':

    parser = beehive.parse.ArgumentParser(('L', 'U', 'beta', 'nt', 'momentum', ))
    parser.add_argument('--Spin', type=int, nargs=2, default=(+1, +1))
    parser.add_argument('--Isospin', type=int, nargs=2, default=(+1, +1))
    args = parser.parse_args()

    # Tuples are preferred.
    args.Spin = (args.Spin[0], args.Spin[1])
    args.Isospin = (args.Isospin[0], args.Isospin[1])

    lattice = beehive.Honeycomb(*args.L) # Instantiate the lattice
    hubbard = beehive.Hubbard(lattice, args.U) # Instantiate the model

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt) # Instantiate the partition function
    print(Z)

    totalMomentum = lattice.momenta[args.momentum]
    C = two_body_correlator(Z, args.Spin, args.Isospin, totalMomentum) # Calculate the two-body correlation matrix

    # Plot the correlator matrix
    for i, j in product(range(C.shape[0]), range(C.shape[1])):
        fig, ax = Z.plot_correlator(C[i,j])
        fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} (I={args.Isospin[0]} S={args.Spin[0]} Iz={args.Isospin[1]} Sz={args.Spin[1]}) P={totalMomentum} p={i}, {j}')

    plt.show()
