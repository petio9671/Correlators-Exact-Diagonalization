import scipy.sparse

#format = 'csc'
#sparse_array = scipy.sparse.csc_array
format = 'csr'
sparse_array = scipy.sparse.csr_array


from beehive.honeycomb import Honeycomb
from beehive.hubbard import Hubbard
from beehive.partition import PartitionFunction
import beehive.cli
