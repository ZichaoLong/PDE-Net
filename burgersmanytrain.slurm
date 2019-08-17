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
viscosity=0.05
momentsparsity=0.001
noise=0.001
data_start_time=1
DEVICES=4
TRAINREPEATSIZE=24
i=0

for sparsity in ${SPARSITY[@]}; do
    basename=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
    for ((trainidx=0;trainidx<$TRAINREPEATSIZE;trainidx++)); do
        DEVICENUM=`expr $i % $DEVICES`
        echo --name=${basename}-id${trainidx} \
            --kernel_size=${kernel_size} --viscosity=${viscosity} --blocks=0-6,9 \
            --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
            --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
            --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1/1600 --hidden_layers=5 \
            --start_noise=${noise} --end_noise=${noise} --scheme=${scheme}
        python train.py \
            --name=${basename}-id${trainidx} \
            --kernel_size=${kernel_size} --viscosity=${viscosity} --blocks=0-6,9 \
            --start_noise=${noise} --end_noise=${noise} \
            --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
            --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
            --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1/1600 --hidden_layers=5 --scheme=${scheme} &
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
wait
