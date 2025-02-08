for ((i=$1;i<=$2;i++));do
    echo make sweep GPUS="device=$i" SWEEP_ID=$3
done