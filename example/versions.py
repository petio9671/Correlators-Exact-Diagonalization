import numpy as np
import scipy as sc

import beehive

dependencies = (('numpy', np.__version__),
                ('scipy', sc.__version__),
                )

longest_name = max((len(d[0]) for d in dependencies))

for name, version in dependencies:
    print(f'{name:{longest_name}}    {version}')
