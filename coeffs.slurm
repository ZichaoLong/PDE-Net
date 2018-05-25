#!/bin/bash
#SBATCH -J expression
#SBATCH --qos=short
#SBATCH -p geforce
#SBATCH --nodelist=gpu01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH -t 12:00:00

cd ~/workspace/expression/code
export MKL_NUM_THREADS=6
ISSPARSE=(sparse)
BLOCK=(2 9 12 15)
TEST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
i=0

for issparse in ${ISSPARSE[@]}; do
    for block in ${BLOCK[@]}; do
        for test in ${TEST[@]}; do
            python _coeffs.py ${test} ${issparse} ${block} &
        done
        wait
        python coeffs.py ${issparse} ${block}
    done
done
