#!/bin/bash
#SBATCH -J eg1
#SBATCH --qos=short
#SBATCH --partition=k80
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:3
#SBATCH -t 12:00:00

cd `pwd`
export MKL_NUM_THREADS=6
DATANAME="reactiondiffusion"
CONSTRAINT=(frozen)
KERNEL_SIZE=(5)
STABLIZE=(0)
SPARSITY=(0 0.005)
MOMENTSPARSITY=(0 0.001)
NOISE=(0.001)
DATA_START_TIME=(0)
DEVICES=4
# SEED=(0 1 2 3 4 5)
# NPSEEDS=(0 1 1 2 2 3)
# TORCHSEEDS=(9 1 8 2 7 3)
i=0
REPEATSIZE=20

for constraint in ${CONSTRAINT[@]}; do
    for kernel_size in ${KERNEL_SIZE[@]}; do
        for stablize in ${STABLIZE[@]}; do
            for sparsity in ${SPARSITY[@]}; do
                for momentsparsity in ${MOMENTSPARSITY[@]}; do
                    for data_start_time in ${DATA_START_TIME[@]}; do
                        for noise in ${NOISE[@]}; do
                            name=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
                            for ((i=0;i<$REPEATSIZE;i++))
                            do
                                DEVICENUM=`expr $i % $DEVICES`
                                echo "repeat batch:$i, select device:$DEVICENUM"
                                python _test.py $name $i cuda:$DEVICENUM &
                                echo $name $i cuda:$DEVICENUM
                                sleep 0.1
                                if [ "$DEVICENUM" -eq `expr $DEVICES - 1` ]
                                then
                                    echo 'wait'
                                    wait
                                fi
                            done
                            wait
                            echo 'aggregate errs'
                            python test.py $name $REPEATSIZE
                        done
                    done
                done
            done
        done
    done
done
wait
