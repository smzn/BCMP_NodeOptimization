#!/bin/bash
#PBS -q OCTMEM
#PBS --group=G15037
#PBS -l elapstim_req=24:00:00
#PBS -M mizuno.shinya@sist.ac.jp
#PBS -m bea
#PBS -b 1                    # 利用するノード数を指定
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=/octfs/work/G15037/u6b126/omp-modules
cd $PBS_O_WORKDIR
export PATH=/octfs/apl/Anaconda3/bin:$PATH
source activate test-env
var=`date +'%Y%m%d%H%M%S'`
N=33
R=5
K=500
node_number=1
core=128
npernode=128
mpirun ${NQSV_MPIOPTS} -np ${core} -npernode ${npernode} -mca btl_tcp_if_include ib0 python3 BCMP_MVA_Computation_v8.py ${N} ${R} ${K} > output/${var}_BCMP_MVA_Computation_v8_${N}_${R}_${K}_${node_number}_${core}_OCTMEM.txt
 
