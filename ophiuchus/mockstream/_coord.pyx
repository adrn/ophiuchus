# coding: utf-8
# cython: boundscheck=True
# cython: debug=True
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Coordinate help for generating mock streams. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

# __all__ = ['streakline_stream']

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef void sat_rotation_matrix(double *w, double *R):
    cdef:
        double x1_norm, x2_norm, x3_norm = 0.
        unsigned int i
        double[::1] x1 = np.zeros(3)
        double[::1] x2 = np.zeros(3)
        double[::1] x3 = np.zeros(3)

    x1[0] = w[0]
    x1[1] = w[1]
    x1[2] = w[2]

    x3[0] = x1[1]*w[2+3] - x1[2]*w[1+3]
    x3[1] = x1[2]*w[0+3] - x1[0]*w[2+3]
    x3[2] = x1[0]*w[1+3] - x1[1]*w[0+3]

    x2[0] = -x1[1]*x3[2] + x1[2]*x3[1]
    x2[1] = -x1[2]*x3[0] + x1[0]*x3[2]
    x2[2] = -x1[0]*x3[1] + x1[1]*x3[0]

    x1_norm = sqrt(x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2])
    x2_norm = sqrt(x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2])
    x3_norm = sqrt(x3[0]*x3[0] + x3[1]*x3[1] + x3[2]*x3[2])

    for i in range(3):
        x1[i] /= x1_norm
        x2[i] /= x2_norm
        x3[i] /= x3_norm

    R[0] = x1[0]
    R[1] = x1[1]
    R[2] = x1[2]
    R[3] = x2[3]
    R[4] = x2[4]
    R[5] = x2[5]
    R[6] = x3[6]
    R[7] = x3[7]
    R[8] = x3[8]

cdef void _to_sat_coords(double *w, double *w_sat, double *R,
                         double *w_prime):
    # Translate to be centered on progenitor
    cdef:
        double[::1] dw = np.zeros(6)
        int i

    for i in range(6):
        dw[i] = w[i] - w_sat[i]

    # Project into new basis
    w_prime[0] = dw[0]*R[0] + dw[1]*R[1] + dw[2]*R[2]
    w_prime[1] = dw[0]*R[3] + dw[1]*R[4] + dw[2]*R[5]
    w_prime[2] = dw[0]*R[6] + dw[1]*R[7] + dw[2]*R[8]

    w_prime[3] = dw[3]*R[0] + dw[4]*R[1] + dw[5]*R[2]
    w_prime[4] = dw[3]*R[3] + dw[4]*R[4] + dw[5]*R[5]
    w_prime[5] = dw[3]*R[6] + dw[4]*R[7] + dw[5]*R[8]

cdef void _from_sat_coords(double *w_prime, double *w_sat, double *R,
                           double *w):
    cdef int i

    # Project back from sat plane
    w[0] = w_prime[0]*R[0] + w_prime[1]*R[3] + w_prime[2]*R[6]
    w[1] = w_prime[0]*R[1] + w_prime[1]*R[4] + w_prime[2]*R[7]
    w[2] = w_prime[0]*R[2] + w_prime[1]*R[5] + w_prime[2]*R[8]

    w[3] = w_prime[3]*R[0] + w_prime[4]*R[3] + w_prime[5]*R[6]
    w[4] = w_prime[3]*R[1] + w_prime[4]*R[4] + w_prime[5]*R[7]
    w[5] = w_prime[3]*R[2] + w_prime[4]*R[5] + w_prime[5]*R[8]

    for i in range(6):
        w[i] = w[i] + w_sat[i]

