for ((i=$1;i<=$2;i++));do
    make sweep GPUS=$i SWEEP_ID=$3
done