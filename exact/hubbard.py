import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from exact import format, sparse_array

class Hubbard:

    def __init__(self, lattice, U):
        self.Lattice = lattice
        self.U = U

        self.Hilbert_space_dimension = 4**self.Lattice.sites

    def _destroy(self, res):
        sigma = sparse_array(np.array([[1,0],[0,-1]], dtype=np.float64))
        sigma4 = sc.sparse.kron(sigma,sigma)

        leftOp_i = sc.sparse.eye(1, dtype=np.float64, format=format)

        destroy = []
        for sitei in range(self.Lattice.sites):
            big_eye_i = sc.sparse.eye(4**(self.Lattice.sites-sitei-1), dtype=np.float64, format=format)
            rightOp = sc.sparse.kron(res, big_eye_i, format=format)
            destroy.append(sc.sparse.kron(leftOp_i, rightOp, format=format))

            leftOp_i = sc.sparse.kron(leftOp_i, sigma4, format=format)

        return destroy

    @cached_property
    def destroy_particle(self):
        res_plus = sparse_array(np.array([[0., 0.], [1., 0.]], dtype=np.float64))
        I = sc.sparse.eye(2, dtype=np.float64, format=format)
        res_p = sc.sparse.kron(res_plus, I, format=format)

        return self._destroy(res_p)

    @cached_property
    def destroy_hole(self):
        res_minus = sparse_array(np.array([[0., 1.], [0., 0.]], dtype=np.float64))
        sigma = sparse_array(np.array([[1,0],[0,-1]], dtype=np.float64))
        res_m = sc.sparse.kron(sigma, res_minus, format=format) # <- I put plus here

        return self._destroy(res_m)

    @property
    def create_particle(self):
        return [o.T.conj() for o in self.destroy_particle]

    @property
    def create_hole(self):
        return [o.T.conj() for o in self.destroy_hole]

    def operator(self, amplitudes, operators):
        matrix = np.zeros_like(self.Hilbert_space_dimension, dtype=complex)

        for amplitude, op in zip(amplitudes, operators):
            matrix += amplitude * op

        return matrix

    @cached_property
    def _KV(self):
        numSites = self.Lattice.sites

        U_half = np.float64(0.5*self.U)

        matrix_shape = (self.Hilbert_space_dimension, self.Hilbert_space_dimension)
        ham_int = sparse_array(matrix_shape, dtype=np.float64)
        ham_kin = sparse_array(matrix_shape, dtype=np.float64)

        res_minus = sparse_array(np.array([[0., 1.], [0., 0.]], dtype=np.float64))
        res_plus = sparse_array(np.array([[0., 0.], [1., 0.]], dtype=np.float64))
        sigma = sparse_array(np.array([[1,0],[0,-1]], dtype=np.float64))
        I = sc.sparse.eye(2, dtype=np.float64, format=format)
        res_p = sc.sparse.kron(res_plus, I, format=format)
        res_m = sc.sparse.kron(sigma, res_minus, format=format)

        sigma4 = sc.sparse.kron(sigma,sigma)
        leftOp_i = sc.sparse.eye(1, dtype=np.float64, format=format)

        for sitei in range(numSites):
            big_eye_i = sc.sparse.eye(4**(numSites-sitei-1), dtype=np.float64, format=format)
            rightOpP = sc.sparse.kron(res_p, big_eye_i, format=format)
            rightOpH = sc.sparse.kron(res_m, big_eye_i, format=format)

            destroyP = sc.sparse.kron(leftOp_i, rightOpP, format=format)
            destroyH = sc.sparse.kron(leftOp_i, rightOpH, format=format)
            leftOp_i = sc.sparse.kron(leftOp_i, sigma4, format=format)

            ham_int += (destroyP.T.conj() @destroyP - destroyH.T.conj()@destroyH)**2

            leftOp_j = sc.sparse.eye(1, dtype=np.float64, format=format)

            for sitej in range(numSites):
                if (sitei <= sitej) and (self.Lattice.hopping[sitei, sitej] != 0):
                    big_eye_j = sc.sparse.eye(4**(numSites-sitej-1), dtype=np.float64, format=format)
                    rightOpP = sc.sparse.kron(res_p, big_eye_j, format=format)
                    rightOpH = sc.sparse.kron(res_m, big_eye_j, format=format)

                    destroyPP = sc.sparse.kron(leftOp_j, rightOpP, format=format)
                    destroyHH = sc.sparse.kron(leftOp_j, rightOpH, format=format)

                    commonOp = (destroyP.T.conj()@destroyPP) - (destroyH.T.conj()@destroyHH)
                    ham_kin -= (commonOp + commonOp.T.conj())*self.Lattice.hopping[sitei, sitej]

                leftOp_j = sc.sparse.kron(leftOp_j, sigma4, format=format)
        return ham_kin, U_half*ham_int#

    def K(self):
        k, v = self._KV
        return k

    def V(self):
        k, v = self._KV
        return v

    @property
    def Hamiltonian(self):
        K, V = self._KV
        return K+V


