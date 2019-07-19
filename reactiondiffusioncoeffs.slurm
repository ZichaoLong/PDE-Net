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
DATANAME="reactiondiffusion"
CONSTRAINT=(2 frozen)
kernel_size=5
stablize=0
SPARSITY=(0.005 0)
momentsparsity=0.001
noise=0.001
data_start_time=0
DEVICES=4
REPEATSIZE=24
i=0

# for sparsity in ${SPARSITY[@]}; do
#     basename=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
#     for ((testidx=0;testidx<$REPEATSIZE;testidx++)); do
#         DEVICENUM=`expr $i % $DEVICES`
#         echo --name=${basename}-test${testidx} \
#             --kernel_size=${kernel_size} --viscosity=0.1 --blocks=0,0,0-6,9,12,15,18 \
#             --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
#             --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
#             --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1e-4 \
#             --start_noise=${noise} --end_noise=${noise}
#         python train.py \
#             --name=${basename}-test${testidx} \
#             --kernel_size=${kernel_size} --viscosity=0.1 --blocks=0,0,0-6,9,12,15,18 \
#             --start_noise=${noise} --end_noise=${noise} \
#             --dataname=$DATANAME --constraint=${constraint} --device=cuda:${DEVICENUM} \
#             --stablize=${stablize} --sparsity=${sparsity} --momentsparsity=${momentsparsity} \
#             --data_start_time=${data_start_time} --dt=1e-2 --max_dt=1e-4 &
#         echo $DEVICENUM
#         sleep 1
#         if [ "$DEVICENUM" -eq `expr $DEVICES - 1` ]
#         then
#             echo 'wait'
#             wait
#         fi
#         let i++
#     done
# done
# wait

BLOCK=(2 9 12 15)

for sparsity in ${SPARSITY[@]}; do
    for constraint in ${CONSTRAINT[@]}; do
        for block in ${BLOCK[@]}; do
            for ((testidx=0;testidx<$REPEATSIZE;testidx++)); do
                basename=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
                name=${basename}-test${testidx}
                python _burgerscoeffs.py ${name} ${block} &
            done
        done
    done
    wait
done
wait
for constraint in ${CONSTRAINT[@]}; do
    for block in ${BLOCK[@]}; do
        for sparsity in ${SPARSITY[@]}; do
            basename=${DATANAME}-${constraint}-stab${stablize}-sparse${sparsity}-msparse${momentsparsity}-datast${data_start_time}-size${kernel_size}-noise${noise}
            python burgerscoeffs.py ${basename} ${REPEATSIZE} ${block} &
        done
    done
done
wait
