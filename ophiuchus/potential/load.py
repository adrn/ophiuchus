# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.utils.data import get_pkg_data_filename
import gary.potential as gp

def load_potential(name, bar=None):
    from .. import potential as op
    pot = gp.load(get_pkg_data_filename('yml/{}.yml'.format(name)), module=op)
    if bar is not None:
        pars = pot.parameters
        pars['bar']['alpha'] = bar['alpha']
        pars['bar']['Omega'] = bar['Omega']
        pot = op.OphiuchusPotential(**pars)
    return pot

# barred_mw = load_potential('barred_mw')
# static_mw = load_potential('static_mw')
