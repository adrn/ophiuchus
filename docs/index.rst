
Running the experiments
=======================

This document describes how to run the experiments in this package. Each
experiment is run over all potential models: the nine barred potentials and
the one static, axisymmetric potential. These potential models are cached
as YAML files in `ophiuchus/potential/yml` and can be loaded by name using,
e.g.,::

    >>> import ophiuchus.potential as op
    >>> potential = op.load_potential('static_mw')

To run the experiments without specifying paths everywhere, you must define
an environment variable `PROJECTSPATH` that specifies the path that contains
the `ophiuchus` project. For example, if you cloned `ophiuchus` to::

    /Users/yourname/projects/ophiuchus

You must set (in your shell)::

    export PROJECTSPATH=/Users/yourname/projects

The results from the experiments will then be saved in::

    /Users/yourname/projects/ophiuchus/results

All of the experiments can be run from inside::

    /Users/yourname/projects/ophiuchus/scripts

so we recommend that you change directories into the scripts path.

Each step below must be repeated for each potential model (e.g., ``static_mw``,
``barred_mw_1``, ``barred_mw_2``, etc.), but we will only show ``static_mw``
below as an example.

1) Fitting orbits in each potential
-----------------------------------

First, we must fit orbits to the Ophiuchus BHB star data in each potential. We
do this with the following script::

    python fit-orbit.py --potential=static_mw -v --mcmc_steps=256 --fixtime

Or, to run using MPI (e.g., on a cluster -- make sure to replace ``<CORES>>``)::

    mpiexec -n <CORES> python fit-orbit.py --potential=static_mw -v --mcmc_steps=256 --fixtime --mpi

The MCMC walkers should be run for at least 512 steps to ensure convergence.

2) Create initial conditions from MCMC samples
----------------------------------------------

After running the MCMC orbit fitting, we next need to generate independent samples
from the chains by thinning the chains. We'll do this using the following script::

    python make-orbitfit-w0.py --potential=static_mw

This script reads the sampler file, estimates the autocorrelation time for each
parameter, then takes every
