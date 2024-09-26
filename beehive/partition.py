import numpy as np
from scipy import linalg
import scipy as sc

from itertools import product
from functools import cached_property

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
from beehive.monitoring import Timed

from beehive import format

class PartitionFunction:
    """This class implements the partition function and correlators.

    Attributes:
        H (obj): Model object
        beta (float): Inverse temperature
        nt (int): Number of timeslices
        delta (float): Delta

    """

    def __init__(self, H, beta, nt, _continuum = 100):
        self.H = H
        self.beta = beta
        self.nt = nt

        self.delta = self.beta / self.nt
        self.taus = np.arange(nt) * self.delta if nt < float('inf') else beta / _continuum * np.arange(_continuum)
        self._timeslices = len(self.taus) # This is an integer even if nt = inf.

    def __str__(self):
        return f'PartitionFunction({self.H}, beta={self.beta}, nt={self.nt})'

    @cached_property
    @Timed(logger.debug)
    def value(self):
        """sparse_array: value of the partiton function"""
        if self.nt == float('inf'):
            return ( sc.sparse.linalg.expm(-self.beta*self.H.Hamiltonian) ).trace()

        return self._transfers[-1].trace()

    @cached_property
    @Timed(logger.debug)
    def _transfers(self):
        """:list of sparse_array: Transfer matrix which is defined by

            T(t) = \product_{t=(0,...,nt} ( exp^{-\delta K} exp^{-\delta V} )

            where K and V are kinetic and interaction terms respetively
            
        """

        if self.nt == float('inf'):
            H = self.H.Hamiltonian
            return [sc.sparse.linalg.expm(-tau*H) for tau in self.taus]

        transfer_matrix = sc.sparse.linalg.expm(-self.delta*self.H.K()) @ sc.sparse.linalg.expm(-self.delta*self.H.V())
        transfers = [sc.sparse.eye(self.H.Hilbert_space_dimension, format=format)]
        for t in range(1,self._timeslices+1):
            transfers.append(transfers[-1] @ transfer_matrix)
        return transfers

    @Timed(logger.debug)
    def correlator(self, sink, source):
        """Calculate descreet or continuum correlation function
            If it is in the continuum it calculates
            C(\tau) = exp^{-(\beta - \tau) H} @ sink @ exp^{-\tau H} @ source

            Otherwise
            C(\tau) = transfer[nt-\tau] @ sink @ transfer[\tau] @ source

            where transfer is calculated by _transfers

        Args:
            sink (sparse_array): Operator at the sink
            source (sparse_array): Operator at the source

        Returns:
            ndarray: Correlation function
        
        """

        if self.nt == float('inf'):
            H = self.H.Hamiltonian
            return np.array([
                ( sc.sparse.linalg.expm(-(self.beta-tau)*H) @ sink @ sc.sparse.linalg.expm(-tau*H) @ source ).trace()
                for tau in self.taus
            ]) / self.value

        c = np.zeros(self._timeslices, dtype=complex)
        for t in range(0, self._timeslices):
            c[t] = (self._transfers[self._timeslices-t] @ sink @ self._transfers[t] @ source).trace()

        return c / self.value

    @Timed(logger.debug)
    def correlator_matrix(self, sink, source):
        """Construct the band correlator matrix

        Args:
            sink (sparse_array): Operator at the sink
            source (sparse_array): Operator at the source

        Returns:
            ndarray: Correlator matrix
        
        """

        I, J = len(sink), len(source)
        correlator = np.zeros((I, J, len(self.taus)), dtype=complex)
        for i, j in product(range(I), range(J)):
            correlator[i,j] = self.correlator(sink[i].T.conj(), source[j])

        return correlator

    def plot_correlator(self, C, axsize=(4, 4), **kwargs):
        """Plot correlator matrix

        Args:
            C (sparse_array): Correlator matrix
            axsize (tuple of int, optional): size of the plot of the elements
            **kwargs

        Returns:
            tuple: fig, ax
        
        """

        fig, ax = plt.subplots(*C.shape[:2],
                               figsize=(axsize[0] * C.shape[0], axsize[1] * C.shape[1]),
                               )
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


