PRECISION="double"
CONSTRAINT="moment"
KERNEL_SIZE=7
MAX_ORDER=2
TASKDESCRIPTOR="nonlinpde"${KERNEL_SIZE}x${KERNEL_SIZE}${CONSTRAINT}${MAX_ORDER}order-${PRECISION}
echo ${TASKDESCRIPTOR}
python learn_singlenonlinear2d.py --kernel_size=${KERNEL_SIZE} --max_order=${MAX_ORDER} --precision=${PRECISION} --constraint=${CONSTRAINT} --taskdescriptor=${TASKDESCRIPTOR}
python nonlinpdetest.py ${TASKDESCRIPTOR}
