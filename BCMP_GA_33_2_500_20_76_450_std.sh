#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID
#PBS --group=G15281
#PBS -m eb
#PBS -M mizuno.shinya@sist.ac.jp
#PBS -l cpunum_job=76
#PBS -l elapstim_req=24:00:00
#------- Program execution -----------
module load BasePy/2021
module load BaseCPU
source /sqfs/work/G15281/v60550/test-env/bin/activate
cd $PBS_O_WORKDIR
mpiexec -n 76 python3 BCMP_GA_Class_std.py 33 2 500 20 76 450 > BCMP_GA_33_2_500_20_76_450_std.txt

