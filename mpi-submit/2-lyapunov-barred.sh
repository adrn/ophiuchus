#!/bin/sh

# Directives
#PBS -N lyapunov-barred
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=08:00:00,mem=8gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V
#PBS -t 1-9

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

module load openmpi/1.6.5-no-ib

# print date and time to file
date

cd /vega/astro/users/amp2217/projects/ophiuchus/scripts/

source activate ophiuchus

python lyapunov.py --potential=barred_mw_$PBS_ARRAYID -v -o

date

#End of script
