import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

import matplotlib.pyplot as plt

from beehive import format

class PartitionFunction:

    def __init__(self, H, beta, nt, _continuum = 100):
        self.H = H
        self.beta = beta
        self.nt = nt

        self.delta = self.beta / self.nt
        self.taus = np.arange(nt) * self.delta if nt < float('inf') else beta / _continuum * np.arange(_continuum)

        

    @cached_property
    def value(self):
        if self.nt == float('inf'):
            return ( sc.sparse.linalg.expm(-self.beta*self.H.Hamiltonian) ).trace()

        transfer_matrix = sc.sparse.linalg.expm(-self.delta*self.H.K().toarray()) @ sc.sparse.linalg.expm(-self.delta*self.H.V().toarray())
        return sc.sparse.linalg.matrix_power(transfer_matrix, self.nt).trace()

    @cached_property
    def _transfers(self):
        transfer_matrix = sc.sparse.linalg.expm(-self.delta*self.H.K()) @ sc.sparse.linalg.expm(-self.delta*self.H.V())
        transfers = [sc.sparse.eye(self.H.Hilbert_space_dimension, format=format)]
        for t in range(1,self.nt+1):
            transfers.append(transfers[-1] @ transfer_matrix)
        return transfers

    def correlator(self, sink, source):
        H = self.H.Hamiltonian

        if self.nt == float('inf'):
            return np.array([
                ( sc.sparse.linalg.expm(-(self.beta-tau)*H) @ sink @ sc.sparse.linalg.expm(-tau*H) @ source ).trace()
                for tau in self.taus
            ]) / self.value

        c = np.zeros(self.nt, dtype=complex)
        for t in range(0, self.nt):
            c[t] = (self._transfers[self.nt-t] @ sink @ self._transfers[t] @ source).trace()

        return c / self.value

    def plot_correlator(self, C):
        fig, ax = plt.subplots(*C.shape[:2])
        style = {
                'marker': '.', 'linestyle': 'none',
                } if self.nt < float('inf') else {
                'marker': 'none',
                }

        for C_sink, ax_sink in zip(C, ax):
            for C_sink_source, ax_sink_source in zip(C_sink, ax_sink):
                ax_sink_source.plot(self.taus[1:], C_sink_source[1:].real, **style, label='real')
                ax_sink_source.plot(self.taus[1:], C_sink_source[1:].imag, **style, label='imaginary')

        ax[0, -1].legend()
        return fig, ax


