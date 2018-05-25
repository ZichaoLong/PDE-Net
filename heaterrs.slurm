#!/bin/bash
#SBATCH -J eg1
#SBATCH --qos=short
#SBATCH --partition=k80
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:3
#SBATCH -t 12:00:00

cd ~/workspace/expression/code
DATANAME="heat"
CONSTRAINT=(frozen)
STABLIZE=(0)
SPARSITY=(0 0.001 0.005)
MOMENTSPARSITY=(0.001)
DATA_START_TIME=(0)
kernel_size=5
DEVICES=5
# SEED=(0 1 2)
# NPSEEDS=(0 1 2)
# TORCHSEEDS=(9 8 7)
i=0
REPEATSIZE=20
for constraint in ${CONSTRAINT[@]}; do
    for stablize in ${STABLIZE[@]}; do
        for sparsity in ${SPARSITY[@]}; do
            for momentsparsity in ${MOMENTSPARSITY[@]}; do
                for data_start_time in ${DATA_START_TIME[@]}; do
                    DEVICENUM=`expr $i % $DEVICES`
                    for ((i=0;i<$REPEATSIZE;i++))
                    do
                        name=${DATANAME}-${constraint}-sparse${sparsity}
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
