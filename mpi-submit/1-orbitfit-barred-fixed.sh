#!/bin/sh

# Directives
#PBS -N orbitfit-barred
#PBS -W group_list=yetiastro
#PBS -l nodes=4:ppn=16,walltime=04:00:00,mem=8gb
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
mpiexec -n 64 python fit-orbit.py --potential=barred_mw_fixed -v --mcmc_steps=1024 --fixtime --mpi -o

# Continue
# mpiexec -n 64 python fit-orbit.py --potential=barred_mw_$PBS_ARRAYID -v --mcmc_steps=512 --fixtime --mpi --continue --mcmc_walkers=64

python make-orbitfit-w0.py --potential=barred_mw_fixed -v -o

date

#End of script
