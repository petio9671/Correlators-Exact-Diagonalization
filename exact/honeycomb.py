import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from exact import format

from matplotlib.patches import Polygon

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

    def __str__(self):
        return f'Honeycomb({self.L[0]}, {self.L[1]})'

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

    def eq_mod_lattice(self, x, y, eps=1e-12):
        diff = x-y

        # Now we try to write diff as integer multiples of b
        denominator = self.unit_cells * (self.a[0,1] * self.a[1,0] - self.a[0,0] * self.a[1, 1])
        m = self.L[1] * (diff[1] * self.a[1,0] - diff[0] * self.a[1,1]) / denominator
        n = self.L[0] * (diff[0] * self.a[0,1] - diff[1] * self.a[0,0]) / denominator

        return (abs(np.round(m)-m)<eps) and (abs(np.round(n)-n)<eps)

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

    @cached_property
    def momenta(self):
        num_momenta = self.unit_cells

        momenta = np.zeros((num_momenta, 2))
        for n in range(self.L[0]):
            for m in range(self.L[1]):
                momenta[n*self.L[1] + m] = self.mod_bz(n*self.b[0]/self.L[0] + m*self.b[1]/self.L[1])
        return momenta

    def mod_bz(self, p):
        # This is a stupid brute-force method that could probably be much smarter.
        in_bz = False
        while not in_bz:
            # Try to make the vector smaller any way you can.
            if ((p - self.b[0])**2).sum() < (p**2).sum():
                p -= self.b[0]
            elif ((p + self.b[0])**2).sum() < (p**2).sum():
                p += self.b[0]
            elif ((p - self.b[1])**2).sum() < (p**2).sum():
                p -= self.b[1]
            elif ((p + self.b[1])**2).sum() < (p**2).sum():
                p += self.b[1]
            # Some points might need a combination of b[0] and b[1] to get smaller.
            elif ((p + self.b[0] + self.b[1])**2).sum() < (p**2).sum():
                p += self.b[0] + self.b[1]
            elif ((p + self.b[0] - self.b[1])**2).sum() < (p**2).sum():
                p += self.b[0] - self.b[1]
            elif ((p - self.b[0] + self.b[1])**2).sum() < (p**2).sum():
                p += - self.b[0] + self.b[1]
            elif ((p - self.b[0] - self.b[1])**2).sum() < (p**2).sum():
                p += - self.b[0] - self.b[1]
            else:
                in_bz = True

        return p

    def eq_mod_BZ(self, k, q, eps=1e-12):
        diff = k-q

        # Now we try to write diff as integer multiples of b
        denominator = self.b[0,1] * self.b[1,0] - self.b[0,0] * self.b[1, 1]
        m = (diff[1] * self.b[1,0] - diff[0] * self.b[1,1]) / denominator
        n = (diff[0] * self.b[0,1] - diff[1] * self.b[0,0]) / denominator

        return (abs(np.round(m)-m)<eps) and (abs(np.round(n)-n)<eps)


    def plot_lattice(self, ax, colors=('red', 'blue')):
        
        ax.scatter(self.positions[0::2, 0], self.positions[0::2, 1], color=colors[0])
        ax.scatter(self.positions[1::2, 0], self.positions[1::2, 1], color=colors[1])
        ax.set_aspect('auto')

    def plot_bz(self, ax, **kwargs):
        momenta = self.momenta
        b = self.b
        boundary = 1./3 * np.array([
            -  b[0]+  b[1],
            -2*b[0]-  b[1],
            -  b[0]-2*b[1],
            +  b[0]-  b[1],
            +2*b[0]+  b[1],
            +  b[0]+2*b[1],
            ])
        ax.add_patch(Polygon( boundary,
            edgecolor='black',
            fill=False,

            ))#fillstyle='none', color='black')
        options = {
                'color': 'black',
                'marker': '.',
                'linestyle': 'none',
                }
        options.update(kwargs)
        ax.plot(momenta[:,0], momenta[:,1], **options)
        ax.set_xlim((-2.5,+2.5))
        ax.set_ylim((-2.5,+2.5))
        ax.axis('off')
        ax.set_aspect(1)
