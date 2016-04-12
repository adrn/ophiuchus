# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import

import os
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    # Get gary path
    import gary
    gary_base_path = os.path.split(gary.__file__)[0]
    gary_path = os.path.join(gary_base_path, 'potential')

    # Get biff path
    import biff
    biff_base_path = os.path.split(biff.__file__)[0]
    biff_incl_path = os.path.join(biff_base_path, "src")
    print(biff_incl_path)

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gary_path)
    cfg['include_dirs'].append(biff_incl_path)
    cfg['sources'].append('ophiuchus/potential/core.pyx')
    cfg['sources'].append('ophiuchus/potential/src/_potential.c')
    cfg['sources'].append(os.path.join(gary_path, 'src', 'cpotential.c'))
    cfg['sources'].append(os.path.join(biff_incl_path, 'bfe.c'))
    cfg['sources'].append(os.path.join(biff_incl_path, 'bfe_helper.c'))
    cfg['libraries'] = ['gsl', 'gslcblas']
    cfg['extra_compile_args'] = ['--std=gnu99']

    return [Extension('ophiuchus.potential._potential', **cfg)]

def get_package_data():
    return {'ophiuchus.potential': ['yml/*.yml']}
