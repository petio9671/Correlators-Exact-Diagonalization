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
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--nt', type=int, default=16)
    parser.add_argument('--versions', default=False, action='store_true')
    parser.add_argument('--momentum', type=int, default=0)
    args = parser.parse_args()

    if args.versions:
        print('numpy', np.__version__)
        print('scipy', sc.__version__)
        exit(0)

    lattice = exact.Honeycomb(*args.L)
    hubbard = exact.Hubbard(lattice, args.U)
    energies, unitary = np.linalg.eigh(hubbard.Hamiltonian.toarray())

    Z_discretized = exact.PartitionFunction(hubbard, args.beta, args.nt)

    momentum = lattice.momenta[args.momentum]
    c_plus  = lattice.fourier(momentum, +1)
    c_minus = lattice.fourier(momentum, -1)

    p_plus  = hubbard.operator(c_plus,  hubbard.destroy_particle)
    p_minus = hubbard.operator(c_minus, hubbard.destroy_particle)

    Cd00 = Z_discretized.correlator(p_plus.T.conj(), p_plus)
    Cd01 = Z_discretized.correlator(p_plus.T.conj(), p_minus)
    Cd10 = Z_discretized.correlator(p_minus.T.conj(), p_plus)
    Cd11 = Z_discretized.correlator(p_minus.T.conj(), p_minus)

    fig, ax = plt.subplots(2,2)

    ax[0,0].plot(Z_discretized.taus, Cd00.real)
    ax[0,1].plot(Z_discretized.taus, Cd01.real, label='real')
    ax[1,0].plot(Z_discretized.taus, Cd10.real)
    ax[1,1].plot(Z_discretized.taus, Cd11.real)

    ax[0,0].plot(Z_discretized.taus, Cd00.imag)
    ax[0,1].plot(Z_discretized.taus, Cd01.imag, label='imaginary')
    ax[1,0].plot(Z_discretized.taus, Cd10.imag)
    ax[1,1].plot(Z_discretized.taus, Cd11.imag)


    ax[0, 1].legend()
    ax[0,0].set_yscale('log')
    ax[1,1].set_yscale('log')

    fig.suptitle(f'{lattice} U={hubbard.U} β={Z_discretized.beta} nt={Z_discretized.nt} p={momentum}')

    plt.show()
