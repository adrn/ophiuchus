# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import

from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    # Get gary path
    import os
    import gary
    gary_base_path = os.path.split(gary.__file__)[0]
    gary_incl_path = os.path.join(gary_base_path, "potential")

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gary_incl_path)
    cfg['sources'].append('ophiuchus/potential/core.pyx')
    cfg['sources'].append('ophiuchus/potential/src/_potential.c')
    cfg['sources'].append(os.path.join(gary_incl_path, '_cbuiltin.c'))
    cfg['libraries'] = ['gsl', 'gslcblas']

    return [Extension('ophiuchus.potential._potential', **cfg)]
