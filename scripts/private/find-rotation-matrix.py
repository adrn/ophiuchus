# coding: utf-8

""" Determine the rotation matrix to convert to Ophiuchus stream coordinates.

* You don't have to run this script! The rotation matrix is provided in
ophiuchus/ophiuchus/coordinates/rotationmatrix.txt and an Astropy coordinate frame to
transform to Ophiuchus coordinates is in ophiuchus/coordinates
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.coordinates.angles import rotation_matrix
import matplotlib.pyplot as pl
import numpy as np
from gala.dynamics import orbitfit

# Project
from ophiuchus.data import OphiuchusData

def main(plot=False):
    all_data = OphiuchusData() # stars from both 2015a and 2015b
    data = OphiuchusData(expr="source == 'Sesar2015a'") # just use the original BHB stars

    # compute the rotation matrix to go from Galacic to Stream coordinates, first without
    #   an extra rotation to put the center at lon=0
    np.random.seed(42)
    R1 = orbitfit.compute_stream_rotation_matrix(data.coord)

    if plot:
        # rotate all data to plot
        rot_rep = orbitfit.rotate_sph_coordinate(all_data.coord, R1)
        pl.figure()
        pl.suptitle("Before rotating in longitude", fontsize=18)
        pl.plot(rot_rep.lon.wrap_at(180*u.deg).degree, rot_rep.lat.degree, ls='none', marker='o', ms=5.)

    rot_rep = orbitfit.rotate_sph_coordinate(data.coord, R1)

    # now rotate by the median longitude to put the 'center' of the stream at lon=0
    extra_rot = np.median(rot_rep.lon.wrap_at(180*u.deg))
    R2 = rotation_matrix(extra_rot, 'z')
    R = np.asarray(R2*R1)

    print("Rotation matrix:")
    for i in range(3):
        print("{:>20.15f} {:>20.15f} {:>20.15f}".format(*R[i]))

    all_rot_rep = orbitfit.rotate_sph_coordinate(all_data.coord, R)
    if plot:
        pl.figure()
        pl.suptitle("After rotating in longitude", fontsize=18)
        pl.plot(all_rot_rep.lon.wrap_at(180*u.deg).degree, all_rot_rep.lat.degree,
                ls='none', marker='o', ms=5.)

        pl.show()

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", dest="plot", action="store_true", default=False,
                        help="Plot or not")

    args = parser.parse_args()
    main(args.plot)
