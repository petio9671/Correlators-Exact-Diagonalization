from itertools import product
import numpy as np
import beehive
import matplotlib.pyplot as plt

import scipy as sc

import logging
logger = logging.getLogger(__name__)

from one_body import one_body_correlator
from two_body import two_body_correlator

class PDF:
    # A context manager which helps us avoid having too many figures open at once.
    # see eg. history.py for use.
    def __init__(self, filename):
        self.filename = filename
        self.pages = None

    def save(self, fig):
        fig.savefig(self.pages, format='pdf')

    def __enter__(self):
        from matplotlib.backends.backend_pdf import PdfPages
        self.pages = PdfPages(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.pages:
            self.pages.close()
            self.pages = None

if __name__ == '__main__':


    parser = beehive.parse.ArgumentParser('L', 'U', 'beta', 'nt', 'momentum', 'species')
    parser.add_argument('--versions', default=False, action='store_true')
    parser.add_argument('pdf', type=str, default='')

    args = parser.parse_args()

    if args.versions:
        print('numpy', np.__version__)
        print('scipy', sc.__version__)
        exit(0)

    lattice = beehive.Honeycomb(*args.L)
    hubbard = beehive.Hubbard(lattice, args.U)

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt)

    with PDF(args.pdf) as pdf:
        for momentum in lattice.momenta:
            for species in ('particle', 'hole'):
                C = one_body_correlator(Z, species, momentum)

                fig,ax = Z.plot_correlator(C)
                ax[0,0].set_yscale('log')
                ax[1,1].set_yscale('log')
                fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} {species} p={momentum}')
                pdf.save(fig)
                plt.close(fig)

        for Spin, Isospin in (
                # ((S, Sz), (I, Iz))
                ((1, +1), (1,+1)),
                ((1, +1), (1,-1)),
                ((1, -1), (1,+1)),
                ((1, -1), (1,-1)),
                # ... and other combinations not yet implemented ...
                ):
            for TotalMomentum in lattice.momenta:
                C = two_body_correlator(Z, Spin, Isospin, TotalMomentum)
                for i, j in product(range(C.shape[0]), range(C.shape[1])):
                    fig, ax = Z.plot_correlator(C[i,j])
                    fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} S={Spin[0]} Sz={Spin[1]} I={Isospin[0]} Iz={Isospin[1]} P={TotalMomentum} p={i}, {j}')

                    pdf.save(fig)
                    plt.close(fig)
