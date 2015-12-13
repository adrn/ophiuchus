#!/bin/sh

# Directives
#PBS -N orbitfit-static
#PBS -W group_list=yetiastro
#PBS -l nodes=4:ppn=16,walltime=02:00:00,mem=8gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

module load openmpi/1.6.5-no-ib

# print date and time to file
date

cd /vega/astro/users/amp2217/projects/ophiuchus/scripts/

source activate ophiuchus

# New run
# mpiexec -n 64 python fit-orbit.py --potential=static_mw -v --mcmc_steps=512 --fixtime --mpi

# Continue
mpiexec -n 64 python fit-orbit.py --potential=static_mw -v --mcmc_steps=512 --fixtime --mpi --continue --mcmc_walkers=64

python make-orbitfit-w0.py --potential=static_mw -v -o

date

#End of script
