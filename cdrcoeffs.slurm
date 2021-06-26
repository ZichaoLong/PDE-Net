#!/bin/bash
#SBATCH -J expression
#SBATCH --qos=short
#SBATCH -p geforce
#SBATCH --nodelist=gpu01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH -t 12:00:00

cd `pwd`
export MKL_NUM_THREADS=6
DATANAME="cdr"
CONSTRAINT=(2)
kernel_size=5
stablize=0
SPARSITY=(0.005)
scheme=upwind
momentsparsity=0.001
viscosity=0.1
noise=0.001
data_start_time=1
DEVICES=4
REPEATSIZE=24
i=0

BLOCK=(6)

for constraint in ${CONSTRAINT[@]}; do
    for sparsity in ${SPARSITY[@]}; do
        for block in ${BLOCK[@]}; do
            echo ${block}
            for ((trainidx=0;trainidx<$REPEATSIZE;trainidx++)); do
                basename=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
                name=${basename}-id${trainidx}
                echo ${name}
                python _coeffs.py ${name} ${block} &
            done
        done
        wait
    done
done
for constraint in ${CONSTRAINT[@]}; do
    for block in ${BLOCK[@]}; do
        for sparsity in ${SPARSITY[@]}; do
            basename=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
            python coeffs.py ${DATANAME} ${basename} ${REPEATSIZE} ${block}
        done
    done
done
