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
SPARSITY=(0 0.005)
momentsparsity=0.001
noise=0.001
data_start_time=1
DEVICES=8
REPEATSIZE=24
i=0

for sparsity in ${SPARSITY[@]}; do
    basename=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
    for ((testidx=0;testidx<$REPEATSIZE;testidx++)); do
        DEVICENUM=`expr $i % $DEVICES`
        echo --name=${basename}-test${testidx} \
            --kernel_size=${kernel_size} \
            --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
            --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
            --data_start_time=${data_start_time} \
            --start_noise=${noise} --end_noise=${noise}
        python train.py \
            --name=${basename}-test${testidx} \
            --kernel_size=${kernel_size} \
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
wait

BLOCK=(2 9 12 15)

for sparsity in ${SPARSITY[@]}; do
    for block in ${BLOCK[@]}; do
        basename=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
        for ((testidx=0;testidx<$REPEATSIZE;testidx++)); do
            name=${basename}-test${testidx}
            python _burgerscoeffs.py ${name} ${block} &
        done
        wait
        python burgerscoeffs.py ${basename} ${REPEATSIZE} ${block}
    done
done
wait
