#!/bin/bash

for i in $(seq 1 9); do
python best-mockstream.py -c ../results/global_mockstream.cfg -v --potential="barred_mw_$i";
done

python best-mockstream.py -c ../results/global_mockstream.cfg -v --potential=static_mw;
