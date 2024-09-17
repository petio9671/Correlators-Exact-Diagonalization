import numpy as np
import exact
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

def one_body_correlator(Z, momentum, operator):

    # fourier amplitudes
    c_plus  = lattice.fourier(momentum, +1)
    c_minus = lattice.fourier(momentum, -1)

    # band operators
    o_plus  = hubbard.operator(c_plus,  operator)
    o_minus = hubbard.operator(c_minus, operator)


    correlator = np.zeros((2,2, len(Z.taus)), dtype=complex)
    correlator[0,0] = Z.correlator(o_plus.T.conj(), o_plus)
    correlator[0,1] = Z.correlator(o_plus.T.conj(), o_minus)
    correlator[1,0] = Z.correlator(o_minus.T.conj(), o_plus)
    correlator[1,1] = Z.correlator(o_minus.T.conj(), o_minus)

    return correlator

if __name__ == '__main__':


    parser = exact.parse.ArgumentParser('L', 'U', 'beta', 'nt', 'momentum', 'species')
    parser.add_argument('--versions', default=False, action='store_true')

    args = parser.parse_args()

    if args.versions:
        print('numpy', np.__version__)
        print('scipy', sc.__version__)
        exit(0)

    lattice = exact.Honeycomb(*args.L)
    hubbard = exact.Hubbard(lattice, args.U)

    Z = exact.PartitionFunction(hubbard, args.beta, args.nt)

    momentum = lattice.momenta[args.momentum]
    operator = hubbard.destroy_particle if (args.species == 'particle') else hubbard.destroy_hole
    correlator = one_body_correlator(Z, momentum, operator)

    fig, ax = plt.subplots(2,2)
    style = {
            'marker': '.', 'linestyle': 'none',
            } if args.nt < float('inf') else {
            'marker': 'none',
            }

    for C_sink, ax_sink in zip(correlator, ax):
        for C_sink_source, ax_sink_source in zip(C_sink, ax_sink):
            ax_sink_source.plot(Z.taus, C_sink_source.real, **style, label='real')
            ax_sink_source.plot(Z.taus, C_sink_source.imag, **style, label='imaginary')

    ax[0, 1].legend()
    ax[0,0].set_yscale('log')
    ax[1,1].set_yscale('log')

    fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} {args.species} p={momentum}')

    plt.show()
