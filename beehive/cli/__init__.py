import argparse

import logging
logger = logging.getLogger(__name__)

from .log import defaults as log_defaults

def defaults():
    r'''
    Provides a list of standard-library ``ArgumentParser`` objects.

    Currently provides defaults from

    * :func:`beehive.cli.log.defaults`
    '''
    return [
            log_defaults(),
            ]

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

class ArgumentParser(argparse.ArgumentParser):
    r'''
    Forwards all arguments, except that it adds :func:`~.cli.defaults` to the `parents`_ option.

    Parameters
    ----------
        *args:
            Forwarded to the standard library's ``ArgumentParser``.
        *kwargs:
            Forwarded to the standard library's ``ArgumentParser``.

    .. _parents: https://docs.python.org/3/library/argparse.html#parents
    '''
    def __init__(self, flags, *args, **kwargs):
        k = {**kwargs}
        if 'parents' in k:
            k['parents'] += defaults()
        else:
            k['parents'] = defaults()
        super().__init__(*args, **k, epilog=f'Built on the beehive library.')
        if 'L' in flags:
            self.add_argument('--L', type=int, nargs=2, default=(1,1), help='Two positive integers setting the honeycomb size')
        if 'U' in flags:
            self.add_argument('--U', type=float, default=1.0, help='Interaction strength')
        if 'beta' in flags:
            self.add_argument('--beta', type=beta, default=3.0, help='Inverse temperature')
        if 'nt' in flags:
            self.add_argument('--nt', type=nt, default=16, help='Positive integer or inf')
        if 'momentum' in flags:
            self.add_argument('--momentum', type=int, default=0, help='An integer into the lattice.momenta; see plot_lattice.py')
        if 'species' in flags:
            self.add_argument('--species', choices=('p', 'h', 'particle', 'hole'), type=species, default='particle')

    def parse_args(self, args=None, namespace=None):
        r'''
        Forwards to the `standard library`_ but logs all the parsed values at the `DEBUG` level.

        .. _standard library: https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        '''

        parsed = super().parse_args(args, namespace)

        for arg in parsed.__dict__:
            logger.debug(f'{arg}: {parsed.__dict__[arg]}')

        return parsed

def W(w):
    r'''
    Allow W to be any integer or float('inf') if w is in {inf, ∞, infinity, infty}.
    '''

    if w in ('inf', '∞', 'infinity', 'infty'):
        return float('inf')

    try:
        return int(w)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'{w} not a definite integer or infinity')

def input_file(module_name):
    r'''
    You pass the module name; the user passes the file name.

    ```
    parser.add_argument('input_file', type=supervillain.cli.input_file('module_name'), default='filename.py')
    ```
    '''

    def curried(file_name):

        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return curried
