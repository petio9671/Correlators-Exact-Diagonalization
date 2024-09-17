import argparse

def beta(s):
    b = float(s)
    assert b >= 0
    return b

def nt(s):
    if 'inf' in s:
        return float('inf')
    n = int(s)
    assert n > 0
    return n

def species(s):
    if s == 'p':
        return 'particle'
    if s == 'h':
        return 'hole'
    return s

def ArgumentParser(*flags):
    parser = argparse.ArgumentParser()
    if 'L' in flags:
        parser.add_argument('--L', type=int, nargs=2, default=(1,1), help='Two positive integers setting the honeycomb size')
    if 'U' in flags:
        parser.add_argument('--U', type=float, default=1.0, help='Interaction strength')
    if 'beta' in flags:
        parser.add_argument('--beta', type=beta, default=3.0, help='Inverse temperature')
    if 'nt' in flags:
        parser.add_argument('--nt', type=nt, default=16, help='Positive integer or inf')
    if 'momentum' in flags:
        parser.add_argument('--momentum', type=int, default=0, help='An integer into the lattice.momenta; see plot_lattice.py')
    if 'species' in flags:
        parser.add_argument('--species', choices=('p', 'h', 'particle', 'hole'), type=species)

    return parser
