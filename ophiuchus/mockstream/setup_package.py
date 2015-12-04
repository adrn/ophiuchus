# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import
import os
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    # Get gary path
    import gary
    gary_base_path = os.path.split(gary.__file__)[0]
    gary_incl_path = os.path.join(gary_base_path, "integrate", "cyintegrators", "dopri")

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gary_incl_path)
    cfg['include_dirs'].append(mac_incl_path)
    cfg['sources'].append('ophiuchus/mockstream/_coord.pyx')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('ophiuchus.mockstream._coord', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gary_incl_path)
    cfg['include_dirs'].append(mac_incl_path)
    cfg['sources'].append('ophiuchus/mockstream/_mockstream.pyx')
    cfg['sources'].append(os.path.join(gary_incl_path,"dop853.c"))
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('ophiuchus.mockstream._mockstream', **cfg))

    return exts

def get_package_data():
    return {'ophiuchus': ['ophiuchus/mockstream/_coord.pxd']}
