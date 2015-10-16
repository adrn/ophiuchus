# coding: utf-8
# cython: boundscheck=True
# cython: debug=True
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Generate mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport M_PI

from gary.potential.cpotential cimport _CPotential
from ._coord cimport (sat_rotation_matrix, to_sat_coords, from_sat_coords,
                      cyl_to_car, car_to_cyl)

# __all__ = ['streakline_stream']

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef extern from "dop853.h":
    ctypedef void (*GradFn)(double *pars, double *q, double *grad) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn)
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f, GradFn gradfunc, double *gpars, unsigned norbits) nogil

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fcn, GradFn gradfunc, double *gpars, unsigned norbits,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont)

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   GradFn func, double *pars, unsigned norbits)
    double six_norm (double *x)

cdef extern from "stdio.h":
    ctypedef struct FILE
    FILE *stdout

cpdef streakline_stream(_CPotential cpotential, double[::1] t, double[:,::1] prog_w,
                        int release_every,
                        double G, double[::1] prog_mass,
                        double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    streakline_stream(cpotential, t, prog_w, release_every, G, prog_mass, atol, rtol, nmax)

    Generate a mock stellar stream using the Streakline method.

    Parameters
    ----------
    cpotential : `gary.potential._CPotential`
        An instance of a ``_CPotential`` representing the gravitational potential.
    t : `numpy.ndarray`
        An array of times. Should have shape ``(ntimesteps,)``.
    prog_w : `numpy.ndarray`
        The 6D coordinates for the orbit of the progenitor system at all times.
        Should have shape ``(ntimesteps,6)``.
    release_every : int
        Release particles at the Lagrange points every X timesteps.
    G : numeric
        The value of the gravitational constant, G, in the unit system used.
    prog_mass : `numpy.ndarray`
        The mass of the progenitor at each time. Should have shape ``(ntimesteps,)``.
    atol : numeric (optional)
        Passed to the integrator. Absolute tolerance parameter. Default is 1E-10.
    rtol : numeric (optional)
        Passed to the integrator. Relative tolerance parameter. Default is 1E-10.
    nmax : int (optional)
        Passed to the integrator.
    """
    cdef:
        int i, j, k # indexing
        int res # result from calling dop853
        int ntimes = t.shape[0] # number of times
        int nsteps = ntimes-1 # number of steps
        int nparticles # total number of test particles released

        unsigned ndim = prog_w.shape[1] # phase-space dimensionality
        unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

        double dt0 = t[1] - t[0] # initial timestep
        double[::1] tmp = np.zeros(3) # temporary array

        double[::1] w_prime = np.zeros(6) # 6-position in sat coords
        double[::1] cyl = np.zeros(6) # 6-position in cyl-sat coords

        # used for figuring out how many orbits to integrate at any given release time
        unsigned this_ndim, this_norbits

        double Om # angular velocity squared
        double d, sigma_r # distance, dispersion in release positions
        double r_tide, menc, f # tidal radius, mass enclosed, f factor

        double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
        double[:,::1] R = np.zeros((3,3)) # rotation matrix

    # figure out how many particles are going to be released into the "stream"
    if nsteps % release_every == 0:
        nparticles = 2 * (nsteps // release_every)
    else:
        nparticles = 2 * (nsteps // release_every + 1)

    # container for only current positions of all particles
    cdef double[::1] w = np.empty(nparticles*ndim)

    # -------

    # copy over initial conditions from progenitor orbit to each streakline star
    i = 0
    for j in range(nsteps):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # now go back to each set of initial conditions and:
    #   - put position at +/- tidal radius along radial vector
    #   - set velocity so angular velocity constant
    i = 0
    for j in range(nsteps):
        if (j % release_every) != 0:
            continue

        # angular velocity
        d = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2])
        Om = ((tmp[1]*tmp[5] - tmp[2]*tmp[4])**2 +
              (tmp[0]*tmp[5] - tmp[2]*tmp[3])**2 +
              (tmp[0]*tmp[4] - tmp[1]*tmp[3])**2) / (d*d)

        # gradient of potential in radial direction
        menc = cpotential._mass_enclosed(t[j], &prog_w[j,0], &eps[0], G)
        f = 1. + cpotential._d2_dr2(t[j], &prog_w[j,0], &eps[0], G) / (Om*Om)
        r_tide = (G*prog_mass[j] / (f*menc))**(1/3.)

        # the rotation matrix to transform from satellite coords to normal
        sat_rotation_matrix(&prog_w[j,0], &R[0,0])

        # eject stars at tidal radius with same angular velocity as progenitor
        cyl[0] = r_tide
        cyl[1] = 0.
        cyl[2] = 0.
        cyl[3] = 0.
        cyl[4] = Om*d
        cyl[5] = 0.
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &prog_w[j,0], &R[0,0],
                        &w[2*i*ndim])

        cyl[0] = r_tide
        cyl[1] = M_PI
        cyl[2] = 0.
        cyl[3] = 0.
        cyl[4] = Om*d
        cyl[5] = 0.
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &prog_w[j,0], &R[0,0],
                        &w[2*i*ndim + ndim])

        i += 1

    i = 1
    for j in range(nsteps):
        if j % release_every == 0:
            this_norbits = 2*i
            this_ndim = ndim * this_norbits
            i += 1

        res = dop853(this_ndim, <FcnEqDiff> Fwrapper,
                     <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), this_norbits,
                     t[j], &w[0], t[j+1], &rtol, &atol, 0, NULL, 0,
                     NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

        if res == -1:
            raise RuntimeError("Input is not consistent.")
        elif res == -2:
            raise RuntimeError("Larger nmax is needed.")
        elif res == -3:
            raise RuntimeError("Step size becomes too small.")
        elif res == -4:
            raise RuntimeError("The problem is probably stff (interrupted).")

    return np.asarray(w).reshape(nparticles, ndim)

# cpdef fardal_stream(_CPotential cpotential, double[::1] t, double[:,::1] prog_w,
#                     int release_every,
#                     double G, double[::1] prog_mass,
#                     double atol=1E-10, double rtol=1E-10, int nmax=0):
#     """
#     fardal_stream(cpotential, t, prog_w, release_every, G, prog_mass, atol, rtol, nmax)

#     Generate a mock stellar stream using the particle-spray method
#     from Fardal et al. (2015).

#     Parameters
#     ----------
#     cpotential : `gary.potential._CPotential`
#         An instance of a ``_CPotential`` representing the gravitational potential.
#     t : `numpy.ndarray`
#         An array of times. Should have shape ``(ntimesteps,)``.
#     prog_w : `numpy.ndarray`
#         The 6D coordinates for the orbit of the progenitor system at all times.
#         Should have shape ``(ntimesteps,6)``.
#     release_every : int
#         Release particles at the Lagrange points every X timesteps.
#     G : numeric
#         The value of the gravitational constant, G, in the unit system used.
#     prog_mass : `numpy.ndarray`
#         The mass of the progenitor at each time. Should have shape ``(ntimesteps,)``.
#     atol : numeric (optional)
#         Passed to the integrator. Absolute tolerance parameter. Default is 1E-10.
#     rtol : numeric (optional)
#         Passed to the integrator. Relative tolerance parameter. Default is 1E-10.
#     nmax : int (optional)
#         Passed to the integrator.
#     """
#     cdef:
#         int i, j, k # indexing
#         int res # result from calling dop853
#         int ntimes = t.shape[0] # number of times
#         int nsteps = ntimes-1 # number of steps
#         int nparticles # total number of test particles released

#         unsigned ndim = prog_w.shape[1] # phase-space dimensionality
#         unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

#         double dt0 = t[1] - t[0] # initial timestep
#         double[::1] tmp = np.zeros(3) # temporary array

#         # used for figuring out how many orbits to integrate at any given release time
#         unsigned this_ndim, this_norbits

#         double Om2 # angular velocity squared
#         double d, sigma_r # distance, dispersion in release positions
#         double r_tide, menc, f # tidal radius, mass enclosed, f factor

#         double[::1] eps = np.zeros(3) # used for 2nd derivative estimation

#     # figure out how many particles are going to be released into the "stream"
#     if nsteps % release_every == 0:
#         nparticles = 2 * (nsteps // release_every)
#     else:
#         nparticles = 2 * (nsteps // release_every + 1)

#     # container for only current positions of all particles
#     cdef double[::1] w = np.empty(nparticles*ndim)

#     # TODO: this should be a class...
#     # ------- Parameters  -------
#     cdef:
#         double kr_mean = 2.
#         double kr_disp = 0.5
#         double kphi = 0.
#         double kz_mean = 0.
#         double kz_disp = 0.5

#         double kvr = 0.
#         double kvt_mean = 0.3
#         double kvt_disp = 0.5
#         double kvz_mean = 0.
#         double kvz_disp = 0.5

#     # copy over initial conditions from progenitor orbit to each streakline star
#     i = 0
#     for j in range(nsteps):
#         if (j % release_every) != 0:
#             continue

#         for k in range(ndim):
#             w[2*i*ndim + k] = prog_w[j,k]
#             w[2*i*ndim + k + ndim] = prog_w[j,k]

#         i += 1

#     # now go back to each set of initial conditions and:
#     #   - put position at +/- tidal radius along radial vector
#     #   - set velocity so angular velocity constant
#     #   - add dispersion if specified
#     i = 0
#     for j in range(nsteps):
#         if (j % release_every) != 0:
#             continue

#         # get angular velocity of the progenitor
#         v_car_to_cyl(&w[2*i*ndim], &w[2*i*ndim + 3], &tmp[0])
#         Om2 = tmp[1]*tmp[1] + tmp[2]*tmp[2]

#         # gradient of potential in radial direction
#         menc = cpotential._mass_enclosed(t[j], &prog_w[j,0], &eps[0], G)
#         f = 1. + cpotential._d2_dr2(t[j], &prog_w[j,0], &eps[0], G) / Om2
#         r_tide = (G*prog_mass[j] / (f*menc))**(1/3.)

#         # -------- Position --------
#         car_to_sph(&w[2*i*ndim], &tmp[0])
#         d = tmp[0]
#         tmp[0] = d + r_tide*np.random.normal(kr_mean, kr_disp)
#         sph_to_car(&tmp[0], &w[2*i*ndim])

#         car_to_sph(&w[2*i*ndim + ndim], &tmp[0])
#         tmp[0] = d - r_tide*np.random.normal(kr_mean, kr_disp)
#         sph_to_car(&tmp[0], &w[2*i*ndim + ndim])

#         # sigmar = pos_disp_fac * r_tide / 1.732 # sqrt(3)
#         # if sigmar > 0:
#         #     for k in range(ndim_2):
#         #         w[2*i*ndim + k] = np.random.normal(w[2*i*ndim + k], sigmar)
#         #         w[2*i*ndim + k + ndim] = np.random.normal(w[2*i*ndim + k + ndim], sigmar)

#         # -------- Velocity --------
#         v_car_to_sph(&w[2*i*ndim], &w[2*i*ndim + ndim_2], &tmp[0])
#         # tmp[0] = np.random.normal(tmp[0], vel_disp) # radial velocity dispersion
#         tmp[1] = tmp[1] * (d+r_tide) / d
#         tmp[2] = tmp[2] * (d+r_tide) / d
#         v_sph_to_car(&w[2*i*ndim], &tmp[0], &w[2*i*ndim + ndim_2])

#         v_car_to_sph(&w[2*i*ndim + ndim], &w[2*i*ndim + ndim + ndim_2], &tmp[0])
#         # tmp[0] = np.random.normal(tmp[0], vel_disp) # radial velocity dispersion
#         tmp[1] = tmp[1] * (d-r_tide) / d
#         tmp[2] = tmp[2] * (d-r_tide) / d
#         v_sph_to_car(&w[2*i*ndim + ndim], &tmp[0], &w[2*i*ndim + ndim + ndim_2])

#         i += 1

#     i = 1
#     for j in range(nsteps):
#         if j % release_every == 0:
#             this_norbits = 2*i
#             this_ndim = ndim * this_norbits
#             i += 1

#         res = dop853(this_ndim, <FcnEqDiff> Fwrapper,
#                      <GradFn>cpotential.c_gradient, &(cpotential._parameters[0]), this_norbits,
#                      t[j], &w[0], t[j+1], &rtol, &atol, 0, NULL, 0,
#                      NULL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt0, nmax, 0, 1, 0, NULL, 0);

#         if res == -1:
#             raise RuntimeError("Input is not consistent.")
#         elif res == -2:
#             raise RuntimeError("Larger nmax is needed.")
#         elif res == -3:
#             raise RuntimeError("Step size becomes too small.")
#         elif res == -4:
#             raise RuntimeError("The problem is probably stff (interrupted).")

#     return np.asarray(w).reshape(nparticles, ndim)
