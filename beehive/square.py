import numpy as np
from scipy import linalg
import scipy as sc
from functools import cached_property

from beehive import format

import logging
logger = logging.getLogger(__name__)

from matplotlib.patches import Polygon

class Square:
    """This class implements the Square lattice.

    Attributes:
        L (ndarray): Size of the lattice
        unit_cells (int): Number of unit cells
        sites (int): Number of sites
        A_integers (ndarray): Integer positions of the unit cells in posiiton space (Same as position of A sites)

    """

    a = np.array([
        [+1, 0],
        [0, +1]
    ])

    b = 2*np.pi * np.array([
        [+1, 0],
        [0, +1]
    ])

    def __init__(self, L0, L1):

        self.L = np.array([L0, L1])

        self.unit_cells = L0 * L1
        self.sites = self.unit_cells

        self.A_integers = np.array([
            [i, j] for i in range(L0) for j in range(L1)
        ])

        self.A_positions = np.einsum('ji,ix->jx', self.A_integers, self.a)

        self.positions = np.zeros((self.sites, 2), dtype=float)
        self.positions = self.A_positions

        # ToDo put labels on special momenta
        momentum_labels = {
            'gamma': np.array([0,0]),
        }

    def __str__(self):
        return f'Square({self.L[0]}, {self.L[1]})'

    @cached_property
    def adjacency_list(self):
        """:ndarray: Site adjacency array"""

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
        """:ndarray: Hopping matrix"""

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
        """Check if 2 lattice sites are coinciding taking into account the lattice structure

        Args:
            x (list of float): First site
            y (list of float): Second site
            eps (float, optional): Tolerance of equality

        Returns:
            bool: True if they coincide, False otherwise
        
        """

        diff = x-y

        # Now we try to write diff as integer multiples of b
        denominator = self.unit_cells * (self.a[0,1] * self.a[1,0] - self.a[0,0] * self.a[1, 1])
        m = self.L[1] * (diff[1] * self.a[1,0] - diff[0] * self.a[1,1]) / denominator
        n = self.L[0] * (diff[0] * self.a[0,1] - diff[1] * self.a[0,0]) / denominator

        return (abs(np.round(m)-m)<eps) and (abs(np.round(n)-n)<eps)

    def fourier(self, momentum, band):
        """Calculate the fourier amplitudes of a given mometum and band

        Args:
            momentum (list of float): Momentum
            band (int): Plus or minus bands (±1)

        Returns:
            amplitudes (ndarray): Fourier amplitudes
        
        """

        if band not in (-1, +1):
            raise ValueError("Only +1 ot -1 values are allowed")

        amplitudes = np.zeros(self.sites, dtype=complex)
        for ip, position in enumerate(self.A_integers):
            aPos = np.einsum('i,ij',position,self.a)

            exp_arg_a = -1j*np.einsum('i,i',momentum, aPos)

            fcoef1 = np.exp(exp_arg_a)/np.sqrt(self.unit_cells)

            amplitudes[ip] = fcoef1

        return amplitudes

    @cached_property
    def momenta(self):
        """:ndarray: Array of momenta"""

        num_momenta = self.unit_cells

        momenta = np.zeros((num_momenta, 2))
        for n in range(self.L[0]):
            for m in range(self.L[1]):
                momenta[n*self.L[1] + m] = self.mod_bz(n*self.b[0]/self.L[0] + m*self.b[1]/self.L[1])
        return momenta

    def mod_bz(self, p):
        """Mod a momentum back into the first BZ

        Args:
            p (ndarray): Momentum that is on the lattice

        Returns:
            p (ndarray): Same mometum but mapped back into the first BZ
        
        """

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
        """Check if 2 momenta are coinciding taking into account the lattice structure

        Args:
            k (list of float): First momentum
            q (list of float): Second momentum
            eps (float, optional): Tolerance of equality

        Returns:
            bool: True if k and q coincide, False otherwise
        
        """

        diff = k-q

        # Now we try to write diff as integer multiples of b
        denominator = self.b[0,1] * self.b[1,0] - self.b[0,0] * self.b[1, 1]
        m = (diff[1] * self.b[1,0] - diff[0] * self.b[1,1]) / denominator
        n = (diff[0] * self.b[0,1] - diff[1] * self.b[0,0]) / denominator

        return (abs(np.round(m)-m)<eps) and (abs(np.round(n)-n)<eps)


    def plot_lattice(self, ax, colors=('red', 'blue')):
        """Plot the lattice

        Args:
            ax (matplotlib.Axes): matplotlib Axes
            colors (str, optional): Colors of the bipartide sites
        
        """
        
        ax.scatter(self.positions[:, 0], self.positions[:, 1], color=colors[0])
        ax.set_aspect('auto')

    def plot_bz(self, ax, **kwargs):
        """Plot the BZ

        Args:
            ax: matplotlib Axes
            **kwargs
        
        """

        momenta = self.momenta
        b = self.b
        boundary = 1./2 * np.array([
            -  b[0] + b[1],
            +  b[0] + b[1],
            +  b[0] - b[1],
            -  b[0] - b[1],
            ])
        ax.add_patch(Polygon( boundary,
            edgecolor='black',
            fill=False,

            ))#fillstyle='none', color='black')
        options = {
                'color': 'red',
                'marker': '.',
                }
        options.update(kwargs)
        ax.scatter(momenta[:,0], momenta[:,1], **options)
        ax.set_xlim((-3.5,+3.5))
        ax.set_ylim((-3.5,+3.5))
        ax.axis('off')
        ax.set_aspect(1)

    def momentum_pairs_totaling(self, total_momentum):
        """Calculate all possible pairs of momenta given a total momentum

        Args:
            total_momentum (ndarray): Total momentum

        Yields:
            tuple (ndarray): Two momenta of that total the input param
        
        """

        for first_momentum in self.momenta:
            second_momentum = self.mod_bz(total_momentum-first_momentum)
            if np.array(tuple(self.eq_mod_BZ(second_momentum, l) for l in self.momenta)).any():
                yield first_momentum, second_momentum

