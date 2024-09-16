import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from exact import format

class Honeycomb:

    a = np.sqrt(3)/2 * np.array([
        [np.sqrt(3), +1],
        [np.sqrt(3), -1]
    ])

    b = 2*np.pi / np.sqrt(3) * np.array([
        [ 1/np.sqrt(3), +1],
        [ 1/np.sqrt(3), -1],
    ])

    def __init__(self, L0, L1):

        self.L = np.array([L0, L1])

        self.unit_cells = L0 * L1
        self.sites_per_cell = 2
        self.sites = self.sites_per_cell * self.unit_cells
        self.between_sites_in_a_cell = np.array([1., 0.])

        self.A_integers = np.array([
            [i, j] for i in range(L0) for j in range(L1)
        ])

        self.A_positions = np.einsum('ji,ix->jx', self.A_integers, self.a)
        self.B_positions = self.A_positions + self.between_sites_in_a_cell

        self.positions = np.zeros((self.sites, 2), dtype=float)
        self.positions[0::2] = self.A_positions
        self.positions[1::2] = self.B_positions

        momentum_labels = {
            'gamma': np.array([0,0]),
        }



    @cached_property
    def adjacency_list(self):
        pos = self.positions
        separations = np.zeros((self.sites, self.sites))
        for i in range(self.sites):
            for j in range(self.sites):
                separations[i,j] = np.sum((pos[i] - pos[j])**2)
        dst = np.round(separations, 0)

        def at(i,j,s):
            idx = i*self.L[1]*2 + j*2 + s
            return idx%(self.sites)
        
        for i in range(self.L[0]):
            adj.append((at(i,0,0), at(i,self.L[1]-1,1)))
        for j in range(self.L[1]):
            adj.append((at(0,j,0), at(self.L[0]-1,j,1)))
        
        adj = np.array(sorted(adj))

        return adj

    @cached_property
    def hopping(self):

        if self.L[0] == 1 and self.L[1] == 1:
            return np.array([[0., 1.], [1., 0.]])

        pos = self.positions
        separations = np.zeros((self.sites, self.sites))
        differ_by_L0= np.zeros((self.sites, self.sites))
        differ_by_L1= np.zeros((self.sites, self.sites))
        for i in range(self.sites):
            for j in range(i, self.sites):
                separations[i,j] = np.sum((pos[i] - pos[j])**2)
                differ_by_L0[i,j] = np.sum((pos[i] - pos[j] + self.L[0]*self.a[0])**2)
                differ_by_L1[i,j] = np.sum((pos[i] - pos[j] + self.L[1]*self.a[1])**2)

        neighbors = np.array((np.round(separations, 0) == 1), dtype=float)
        neighbors += np.array((np.round(differ_by_L0, 0) == 1), dtype=float)
        neighbors += np.array((np.round(differ_by_L1, 0) == 1), dtype=float)

        return neighbors + neighbors.T

    def neighbors(self, site):
        ...

    def fourier(self, momentum, band):
        '''
        band in ±1
        '''

        fk = np.exp(1j*momentum[0]) + 2*np.exp(-1j*momentum[0]/2)*np.cos(np.sqrt(3)*momentum[1]/2)
        exp_theta = np.exp(-1j*np.angle(fk))

        amplitudes = np.zeros(self.sites, dtype=complex)
        for ip, position in enumerate(self.A_integers):
            aPos = np.einsum('i,ij',position,self.a)
            bPos = np.einsum('i,ij',position,self.a) + self.between_sites_in_a_cell

            exp_arg_a = -1j*np.einsum('i,i',momentum, aPos)
            exp_arg_b = -1j*np.einsum('i,i',momentum, bPos)

            fcoef1 = band*exp_theta*np.exp(exp_arg_a)/np.sqrt(self.unit_cells*2)
            fcoef2 = np.exp(exp_arg_b)/np.sqrt(self.unit_cells*2)

            amplitudes[self.sites_per_cell * ip] = fcoef1
            amplitudes[self.sites_per_cell * ip + 1] = fcoef2

        return amplitudes

    def momenta(self):
        num_momenta = self.unit_cells

        lattice_vec = (2*np.pi/np.sqrt(3))*np.array([[1/np.sqrt(3), 1], [1/np.sqrt(3), -1]])
        momenta = np.zeros((num_momenta, 2))
        for n in range(self.L[0]):
            for m in range(self.L[1]):
                momenta[n*self.L[1] + m] = n*self.b[0]/self.L[0] + m*self.b[1]/self.L[1]
        return momenta

    def eq_mod_lattice(self, x, y):
        ...

    def eq_mod_BZ(self, k, q):
        ...


    def plot_lattice(self, ax, colors=('red', 'blue')):
        
        ax.scatter(self.positions[0::2, 0], self.positions[0::2, 1], color=colors[0])
        ax.scatter(self.positions[1::2, 0], self.positions[1::2, 1], color=colors[1])
        ax.set_aspect('auto')

    def plot_bz(self, ax, **kwargs):
        pass
