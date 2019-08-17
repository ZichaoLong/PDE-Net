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
DATANAME="cdr"
CONSTRAINT=(frozen)
kernel_size=5
stablize=0
SPARSITY=(0)
SCHEME=(central upwind)
momentsparsity=0.001
viscosity=0.1
noise=0.001
data_start_time=1
DEVICES=4
REPEATSIZE=20
i=0

for constraint in ${CONSTRAINT[@]}; do
    for scheme in ${SCHEME[@]}; do
        for sparsity in ${SPARSITY[@]}; do
            # for ((testidx=0;testidx<24;testidx++)) do
                name=${DATANAME}-${constraint}-${scheme}-sparse${sparsity}-noise${noise}
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
            # done
        done
    done
done
wait
