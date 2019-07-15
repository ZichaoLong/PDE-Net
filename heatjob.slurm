#!/bin/bash
#SBATCH -J expression
#SBATCH --qos=short
#SBATCH -p geforce
#SBATCH --nodelist=gpu01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH -t 6:00:00

cd `pwd`
export MKL_NUM_THREADS=6
DATANAME="heat"
CONSTRAINT=(frozen 2)
KERNEL_SIZE=(5)
STABLIZE=(0)
SPARSITY=(0 0.005)
MOMENTSPARSITY=(0 0.001)
NOISE=(0.001)
DATA_START_TIME=(0)
DEVICES=4
# SEED=(0 1 2)
# NPSEEDS=(0 1 2)
# TORCHSEEDS=(9 8 7)
i=0

for constraint in ${CONSTRAINT[@]}; do
    for kernel_size in ${KERNEL_SIZE[@]}; do
        for stablize in ${STABLIZE[@]}; do
            for sparsity in ${SPARSITY[@]}; do
                for momentsparsity in ${MOMENTSPARSITY[@]}; do
                    for data_start_time in ${DATA_START_TIME[@]}; do
                        for noise in ${NOISE[@]}; do
                            DEVICENUM=`expr $i % $DEVICES`
                            echo --name=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise} \
                                --channel_names=u --viscosity=0.1 --kernel_size=${kernel_size} \
                                --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                                --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
                                --data_start_time=${data_start_time} \
                                --start_noise=${noise} --end_noise=${noise}
                            python train.py \
                                --name=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise} \
                                --channel_names=u --viscosity=0.1 --kernel_size=${kernel_size} \
                                --start_noise=${noise} --end_noise=${noise} \
                                --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                                --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
                                --data_start_time=${data_start_time} &
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
            done
        done
    done
done
wait
