#!/bin/bash
#SBATCH -J expression
#SBATCH --qos=short
#SBATCH -p geforce
#SBATCH --nodelist=gpu01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH -t 6:00:00

cd ~/workspace/expression/code
export MKL_NUM_THREADS=6
DATANAME="heat"
constraint="2"
STABLIZE=(0)
SPARSITY=(0 0.001 0.005)
MOMENTSPARSITY=(0.001)
DATA_START_TIME=(0)
kernel_size=5
DEVICES=3
# SEED=(0 1 2)
# NPSEEDS=(0 1 2)
# TORCHSEEDS=(9 8 7)
i=0

for stablize in ${STABLIZE[@]}; do
    for sparsity in ${SPARSITY[@]}; do
        for momentsparsity in ${MOMENTSPARSITY[@]}; do
            for data_start_time in ${DATA_START_TIME[@]}; do
                DEVICENUM=`expr $i % $DEVICES`
                echo --name=${DATANAME}-${constraint}-stab${stablize}-size${kernel_size}-seed$NPSEED-$TORCHSEED --kernel_size=${kernel_size} \
                    --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                    --stablize=${stablize} --data_start_time=0 --channel_names=u --viscosity=0.1 --batch_size=7
                python train.py \
                    --name=${DATANAME}-${constraint}-sparse${sparsity} --kernel_size=${kernel_size} \
                    --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
                    --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} --data_start_time=${data_start_time} --channel_names=u \
                    --viscosity=0.1 --batch_size=7 --kernel_size=${kernel_size} &
                    # --npseed=$NPSEED --torchseed=$TORCHSEED
                echo $DEVICENUM
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
wait
