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
DATANAME="burgers"
constraint=2
kernel_size=5
stablize=0
SPARSITY=(0)
scheme=upwind
momentsparsity=0.001
viscosity=0.05
noise=0.001
data_start_time=1
DEVICES=4
TRAINREPEATSIZE=24
i=0

BLOCK=(6)

for sparsity in ${SPARSITY[@]}; do
    for block in ${BLOCK[@]}; do
        echo ${block}
        basename=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
        for ((trainidx=0;trainidx<$TRAINREPEATSIZE;trainidx++)); do
            name=${basename}-id${trainidx}
            echo $name
            python _coeffs.py ${name} ${block} &
        done
        wait
        python coeffs.py ${DATANAME} ${basename} ${TRAINREPEATSIZE} ${block}
    done
done
wait
