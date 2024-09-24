import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from beehive import format, sparse_array

class Hubbard:
    """This class implements the Hubbard model.

    Attributes:
        Lattice (obj): Lattice on which we implement the model
        U (float): Interaction strength
        Hilbert_space_dimension (int): size of the Hilbert space

    """

    def __init__(self, lattice, U):
        self.Lattice = lattice
        self.U = U

        self.Hilbert_space_dimension = 4**self.Lattice.sites

    def __str__(self):
        return f'Hubbard({self.Lattice}, U={self.U})'

    def _destroy(self, res):
        """Calculate the destruction operator of a particle/hole.
            This is done by separating the Fock space into the tensor products.

            We define for a single site
                p = \sigma \otimes I_2

                h = P_2 \otimes \sigma

            where P_2 is a 2x2 parity operator. We use parity operator because
            this way we account for the anti-commutation relations.
            
            We then expand this definition to 2 sites

                p_x = p \otimes I_4

                p_y = P_4 \otimes p

                h_x = h \otimes I_4

                h_y = P_4 \otimes h

            where P_4 is now a 4x4 parity operator (P_4 = P_2 \otimes P_2)

            And as a last step we can expand it to any number of sites...


            This method is very similar to what they have done here 
            https://physics.stackexchange.com/questions/457468/what-is-difference-between-fermions-and-spins.
            

        Args:
            res (sparse_array): 2x2 matrix that defines if it is a particle or a hole

        Returns:
            sparse_array: Destroy operator
        
        """

        sigma = sparse_array(np.array([[1,0],[0,-1]], dtype=np.float64))
        sigma4 = sc.sparse.kron(sigma,sigma)

        leftOp_i = sc.sparse.eye(1, dtype=np.float64, format=format)

        destroy = []
        # We have to repeat for every site of the lattice. That is why there exact solutions scale poorly 4^sites
        for sitei in range(self.Lattice.sites):
            # To get better speeds we calculate I_n in one go by matching the needed size n to with the size of leftOp_i operator
            big_eye_i = sc.sparse.eye(4**(self.Lattice.sites-sitei-1), dtype=np.float64, format=format)
            rightOp = sc.sparse.kron(res, big_eye_i, format=format)
            destroy.append(sc.sparse.kron(leftOp_i, rightOp, format=format))

            leftOp_i = sc.sparse.kron(leftOp_i, sigma4, format=format)

        return destroy

    @cached_property
    def destroy_particle(self):
        """:list of sparse_array: Destroy particle operator in positions space"""

        res_plus = sparse_array(np.array([[0., 0.], [1., 0.]], dtype=np.float64))
        I = sc.sparse.eye(2, dtype=np.float64, format=format)
        res_p = sc.sparse.kron(res_plus, I, format=format)

        return self._destroy(res_p)

    @cached_property
    def destroy_hole(self):
        """:list of sparse_array: Destroy hole operator in positions space"""

        res_minus = sparse_array(np.array([[0., 1.], [0., 0.]], dtype=np.float64))
        sigma = sparse_array(np.array([[1,0],[0,-1]], dtype=np.float64))
        res_m = sc.sparse.kron(sigma, res_minus, format=format)

        return self._destroy(res_m)

    @property
    def create_particle(self):
        """:list of sparse_array: Create particle operator in positions space"""

        return [o.T.conj() for o in self.destroy_particle]

    @property
    def create_hole(self):
        """:list of sparse_array: Create hole operator in positions space"""

        return [o.T.conj() for o in self.destroy_hole]

    def operator(self, amplitudes, operators):
        """Transform an operator

        Args:
            amplitudes (ndarray): Amplitudes that are used for the linear combination
            operators (list of sparse_array): Operator to be transformed

        Returns:
            sparse_array: Transformed operator
        
        """

        matrix = np.zeros_like(self.Hilbert_space_dimension, dtype=complex)

        for amplitude, op in zip(amplitudes, operators):
            matrix += amplitude * op

        return matrix

    @cached_property
    def _KV(self):
        """Calculate the kinetic and interaction terms of the Hubbard Hamiltonian.
            This implementation is similar to how the _destroy method is calculated. For details
            check _destroy method's documentation.            

        Returns:
            tuple (sparse_array): Kinetic term, Interaction term
        
        """

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
                # Only calculate the upper triangular part of the hopping matrix 
                # and non-zero elements (hopping matrix is symmetric)
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
        """:sparse_array: Kinetic term of the Hubbard Hamiltonian"""

        k, v = self._KV
        return k

    def V(self):
        """:sparse_array: Interaction term of the Hubbard Hamiltonian"""

        k, v = self._KV
        return v

    @property
    def Hamiltonian(self):
        """:sparse_array: Hubbard Hamiltonian at half-filling"""

        K, V = self._KV
        return K+V


