extern double wang_zhao_bar_value(double t, double *pars, double *q, int n_dim);
extern void wang_zhao_bar_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double wang_zhao_bar_density(double t, double *pars, double *q, int n_dim);

extern double ophiuchus_value(double t, double *pars, double *q, int n_dim);
extern void ophiuchus_gradient(double t, double *pars, double *q, int n_dim, double *grad);
extern double ophiuchus_density(double t, double *pars, double *q, int n_dim);


extern double henon_heiles_value(double t, double *pars, double *q, int n_dim);
extern void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad);
