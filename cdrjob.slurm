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
SPARSITY=(0.005 0)
SCHEME=(upwind central)
momentsparsity=0.001
viscosity=0.1
noise=0.001
data_start_time=1
DEVICES=4
REPEATSIZE=24
i=0

for constraint in ${CONSTRAINT[@]}; do
    for scheme in ${SCHEME[@]}; do
        for sparsity in ${SPARSITY[@]}; do
            name=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
            DEVICENUM=`expr $i % $DEVICES`
            echo --name=${name} \
                --kernel_size=${kernel_size} --viscosity=${viscosity} --blocks=0,0,0-6,9,12,15,18,21,24,27,30,35,40 \
                --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
                --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1e-4 --hidden_layers=5 \
                --start_noise=${noise} --end_noise=${noise}
            python train.py \
                --name=${name} \
                --kernel_size=${kernel_size} --viscosity=${viscosity} --blocks=0,0,0-6,9,12,15,18,21,24,27,30,35,40 \
                --start_noise=${noise} --end_noise=${noise} \
                --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
                --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1e-4 --hidden_layers=5 &
            echo $DEVICENUM
            sleep 1
            if [ "$DEVICENUM" -eq `expr $DEVICES - 1` ]
            then
                echo 'wait'
                wait
            fi
            let i++
        done
    done
done
wait
