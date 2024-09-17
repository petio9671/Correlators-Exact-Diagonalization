import scipy.sparse

#format = 'csc'
#sparse_array = scipy.sparse.csc_array
format = 'csr'
sparse_array = scipy.sparse.csr_array


from exact.honeycomb import Honeycomb
from exact.hubbard import Hubbard
from exact.partition import PartitionFunction
import exact.parse
