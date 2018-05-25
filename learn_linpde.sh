PRECISION="double"
CONSTRAINT="moment"
KERNEL_SIZE=5
MAX_ORDER=4
DT=0.015
NOISE_LEVEL=0.015
TASKDESCRIPTOR="linpde"${KERNEL_SIZE}x${KERNEL_SIZE}${CONSTRAINT}${MAX_ORDER}order${DT}dt${NOISE_LEVEL}noise-${PRECISION}
echo ${TASKDESCRIPTOR}
python learn_variantcoelinear2d.py --precision=${PRECISION} --taskdescriptor=${TASKDESCRIPTOR} --constraint=${CONSTRAINT} --kernel_size=${KERNEL_SIZE} --max_order=${MAX_ORDER} --dt=${DT} --start_noise_level=${NOISE_LEVEL} --end_noise_level=${NOISE_LEVEL}
python linpdetest.py ${TASKDESCRIPTOR}
