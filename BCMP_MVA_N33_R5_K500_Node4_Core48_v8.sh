#!/bin/bash
#PBS -q OCTOPUS
#PBS -l cpunum_job=24,elapstim_req=24:00:00
#PBS -M mizuno.shinya@sist.ac.jp
#PBS -m bea
#PBS -T intmpi               # Intel MPI 実行時に必須DBG
#PBS -b 4                    # 利用するノード数を指定
#PBS -v OMP_NUM_THREADS=24   # 自動並列化/OpenMPプロセス数(1ノード内の自動並列化/OpenMPプロセス数)
cd $PBS_O_WORKDIR
var=`date +'%Y%m%d%H%M%S'`
N=33
R=5
K=500
node_number=4
core=48
mpirun ${NQSII_MPIOPTS} -np ${core} python3 BCMP_MVA_Computation_v8.py ${N} ${R} ${K} > output/${var}_BCMP_MVA_Computation_v8_${N}_${R}_${K}_${node_number}_${core}.txt 
