cdef void sat_rotation_matrix(double *w, double *R)

cdef void to_sat_coords(double *w, double *w_sat, double *R,
                        double *w_prime)

cdef void from_sat_coords(double *w_prime, double *w_sat, double *R,
                          double *w)
