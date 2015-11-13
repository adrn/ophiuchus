# coding: utf-8

""" Plotting utilities """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as pl
import numpy as np
import gary.coordinates as gc

# Project
from . import galactocentric_frame, vcirc, vlsr
from .coordinates import Ophiuchus

__all__ = ['plot_data', 'plot_orbit']

def plot_data(ophdata,
              phi1_lims=[-10.,10.]*u.deg, phi2_lims=[-10.,10.]*u.deg, distance_lims=[5.5,10]*u.kpc,
              mul_lims=[-12,0]*u.mas/u.yr, mub_lims=[-4,12]*u.mas/u.yr, vr_lims=[215,335]*u.km/u.s,
              fig=None):
    """
    TODO!
    """

    if fig is None:
        fig,_axes = pl.subplots(2,3,figsize=(12,8),sharex=True)
    else:
        _axes = fig.axes
    axes = np.ravel(_axes)
    axes[-1].set_visible(True) # HACK because matplotlib bbox issues with invisible plots

    # plot the data points
    style = dict(marker='o', ls='none', ecolor='#666666', ms=3.)
    x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).to(phi1_lims.unit).value
    axes[0].errorbar(x, ophdata.coord_oph.phi2.to(phi2_lims.unit).value, 1E-10*x, **style)
    axes[0].set_ylim(*phi2_lims.value)

    axes[1].errorbar(x, ophdata.coord.distance.to(distance_lims.unit).value,
                     ophdata.coord_err['distance'].to(distance_lims.unit).value, **style)
    axes[1].set_ylim(*distance_lims.value)

    for i,k in enumerate(ophdata.veloc.keys()):
        lims = eval("{}_lims".format(k))
        axes[i+2].errorbar(x, ophdata.veloc[k].to(lims.unit).value,
                           ophdata.veloc_err[k].to(lims.unit).value, **style)
        axes[i+2].set_ylim(*lims.value)

    # bottom axis label
    axes[2].set_xlabel(r'$\phi_1$ [deg]')
    axes[3].set_xlabel(r'$\phi_1$ [deg]')
    axes[4].set_xlabel(r'$\phi_1$ [deg]')

    # vertical axis labels
    axes[0].set_ylabel(r'$\phi_2$ [deg]')
    axes[1].set_ylabel(r'$d$ [kpc]')
    axes[2].set_ylabel(r'$\mu_l$ [mas yr$^{-1}$]')
    axes[3].set_ylabel(r'$\mu_b$ [mas yr$^{-1}$]')
    axes[4].set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]')

    # set all phi1 lims
    axes[0].set_xlim(*phi1_lims.value)
    fig.tight_layout()
    axes[-1].set_visible(False) # HACK because matplotlib bbox issues with invisible plots

    return fig

def plot_orbit(orbit_w, ophdata,
               phi1_lims=[-10.,10.]*u.deg, phi2_lims=[-10.,10.]*u.deg, distance_lims=[5.5,10]*u.kpc,
               mul_lims=[-12,0]*u.mas/u.yr, mub_lims=[-4,12]*u.mas/u.yr, vr_lims=[215,335]*u.km/u.s,
               fig=None):
    """
    TODO!
    """

    if fig is None:
        fig,_axes = pl.subplots(2,3,figsize=(12,8),sharex=True)
    else:
        _axes = fig.axes
    axes = np.ravel(_axes)
    axes[-1].set_visible(True) # HACK because matplotlib bbox issues with invisible plots

    # plot the orbit
    style = dict(marker=None, ls='-')
    w_coord = galactocentric_frame.realize_frame(coord.CartesianRepresentation(orbit_w.T[:3]*u.kpc))\
                                  .transform_to(coord.Galactic)
    w_oph = w_coord.transform_to(Ophiuchus)
    w_vel = gc.vgal_to_hel(w_coord, orbit_w.T[3:]*u.kpc/u.Myr,
                           galactocentric_frame=galactocentric_frame,
                           vcirc=vcirc, vlsr=vlsr)

    x = w_oph.phi1.wrap_at(180*u.deg).to(phi1_lims.unit).value
    axes[0].plot(x, w_oph.phi2.to(phi2_lims.unit).value, **style)
    axes[1].plot(x, w_coord.distance.to(distance_lims.unit).value, **style)

    for i,k in enumerate(ophdata.veloc.keys()):
        lims = eval("{}_lims".format(k))
        axes[i+2].plot(x, w_vel[i].to(lims.unit).value, **style)

    # bottom axis label
    axes[2].set_xlabel(r'$\phi_1$ [deg]')
    axes[3].set_xlabel(r'$\phi_1$ [deg]')
    axes[4].set_xlabel(r'$\phi_1$ [deg]')

    # vertical axis labels
    axes[0].set_ylabel(r'$\phi_2$ [deg]')
    axes[1].set_ylabel(r'$d$ [kpc]')
    axes[2].set_ylabel(r'$\mu_l$ [mas yr$^{-1}$]')
    axes[3].set_ylabel(r'$\mu_b$ [mas yr$^{-1}$]')
    axes[4].set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]')

    # set all phi1 lims
    axes[0].set_xlim(*phi1_lims.value)
    fig.tight_layout()
    axes[-1].set_visible(False) # HACK because matplotlib bbox issues with invisible plots

    return fig