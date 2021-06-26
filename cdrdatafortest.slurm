#!/bin/bash
#SBATCH -J eg1
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00

cd `pwd`
name='cdr-noise0.001'
DEVICES=4
REPEATSIZE=20
for ((i=0;i<$REPEATSIZE;i++))
do
    DEVICENUM=`expr $i % $DEVICES`
    j=`expr $i + 1`
    NPSEED=`expr $j \* 10`
    TORCHSEED=`expr $j \* 100`
    echo "repeat batch:$i, select device:$DEVICENUM"
    echo "NPSEED: $NPSEED, TORCHSEED: $TORCHSEED"
    python _test_data_gen.py $name $i cuda:$DEVICENUM $NPSEED $TORCHSEED &
    sleep 0.1
    if [ "$DEVICENUM" -eq `expr $DEVICES - 1` ]
    then
        echo 'wait'
        wait
    fi
done
wait
