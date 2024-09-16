import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from exact import format

class PartitionFunction:

    def __init__(self, H, beta, nt, _continuum = 500):
        self.H = H
        self.beta = beta
        self.nt = nt

        self.delta = self.beta / self.nt
        self.taus = np.arange(nt) * self.delta if nt < float('inf') else np.linspace(0, beta, _continuum)

        

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


