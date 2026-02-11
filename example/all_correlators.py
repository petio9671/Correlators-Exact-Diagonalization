from itertools import product
import h5py
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

    parser = beehive.cli.ArgumentParser(('L', 'U', 'beta', 'nt'))
    parser.add_argument('pdf', type=str, default='')
    args = parser.parse_args()

    lattice = beehive.Honeycomb(*args.L) # Instantiate the lattice
    # lattice = beehive.Square(*args.L) # Instantiate the lattice
    hubbard = beehive.Hubbard(lattice, args.U) # Instantiate the Hubbard model

    Z = beehive.PartitionFunction(hubbard, args.beta, args.nt) # Instantiate the partition function
    logger.info(Z)


    # Loop through everything that we can calculate with this system and model and write to a file
    with PDF(args.pdf) as pdf, h5py.File(f'U{hubbard.U}_B{Z.beta}_Nt{Z.nt}_all_correlators.h5', 'w') as fl:
        for species in ('particle', 'hole'):
            fl.create_group(f"{species}")
            for momentum in lattice.momenta:
                C = one_body_correlator(Z, species, momentum)

                fig,ax = Z.plot_correlator(C)
                ax[0,0].set_yscale('log')
                ax[1,1].set_yscale('log')
                fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} {species} p={momentum}')
                fig.tight_layout()
                pdf.save(fig)

                fl[f"{species}"].create_dataset(f"P={momentum}", data=C)
        for Spin, Isospin, TotalMomentum in product(
                ((0, 0), (1, +1), (1, 0), (1, -1)),
                ((0, 0), (1, +1), (1, 0), (1, -1)),
                lattice.momenta,
                ):
            C = two_body_correlator(Z, Spin, Isospin, TotalMomentum)
            print(f"Calculated two-body correlator for (S, Sz) = {Spin}, (I, Iz) = {Isospin}, P = {TotalMomentum}")
            fl.create_group(f"TwoBody/I={Isospin[0]}_S={Spin[0]}_Iz={Isospin[1]}_Sz={Spin[1]}/P={TotalMomentum}")
            for i, j in product(range(C.shape[0]), range(C.shape[1])):
                fig, ax = Z.plot_correlator(C[i,j])
                fig.suptitle(f'{lattice} U={hubbard.U} β={Z.beta} nt={Z.nt} (I={Isospin[0]} S={Spin[0]} Iz={Isospin[1]} Sz={Spin[1]}) P={TotalMomentum} p={i}, {j}')
                fig.tight_layout()

                pdf.save(fig)

                fl[f"TwoBody/I={Isospin[0]}_S={Spin[0]}_Iz={Isospin[1]}_Sz={Spin[1]}/P={TotalMomentum}"].create_dataset(f"p={i}, {j}", data=C[i,j])
