# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os

# Third-party
import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u

# Project
from ..data import OphiuchusData
from ..plot import plot_data_orbit, plot_data_stream

# def test_data():
#     ophdata = OphiuchusData()
#     fig = plot_data_orbit(ophdata)
#     fig.axes[0].set_xlim(-8,1)
#     fig.axes[0].set_ylim(-2,2)
#     pl.show()

def test_data(tmpdir):
    ophdata = OphiuchusData()

    fig = plot_data_orbit(ophdata)
    fig.savefig(os.path.join(str(tmpdir),"data_orbit.png"))

    fig = plot_data_stream(ophdata)
    fig.savefig(os.path.join(str(tmpdir),"data_stream.png"))

