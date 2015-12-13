#!/bin/sh

# Directives
#PBS -N mockstream
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=16,walltime=04:00:00,mem=8gb
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

mpiexec -n 16 python mockstreamgrid.py --potential=static_mw -c ../../global_mockstream.cfg --mpi

python plot-mockstream.py --potential=static_mw -c ../../global_mockstream.cfg -v

date

#End of script
