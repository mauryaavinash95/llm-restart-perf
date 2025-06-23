#!/bin/bash -l
### PBS -l nodes=70:system=polaris
### PBS -l walltime=01:00:00
### PBS -q debug-scaling
### PBS -A VeloC
### PBS -l filesystems=home:grand
echo "Submitted job"
NNODES=$(wc -l < $COBALT_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"
source ~/.bash_profile
dlconda
cd ~/dl-io/DeepSpeed/
rm -rf /local/scratch/*
cd ~/


set_model_size() {
    model_size=$1
    if [[ $model_size == 1 ]]; then
        echo "================== 1.3B OPT model (1 node)"
        declare -g m=1
        declare -g H=2048
        declare -g F=8192
        declare -g N=24
        declare -g L=32
        declare -g U=2048
        declare -g S=8
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1  
        declare -g G=100000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 3 ]]; then
        echo "================== 3B BLOOM model (1 node)"
        declare -g m=3
        declare -g H=2560
        declare -g F=8192
        declare -g N=30
        declare -g L=32
        declare -g U=2048
        declare -g S=8
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=1
        declare -g R=1
        declare -g P=1
        declare -g G=10000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 7 ]]; then
        echo "================== 7B LLAMA2 (1 node)"
        declare -g m=7
        declare -g H=4096
        declare -g F=11008
        declare -g N=32
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=4
        declare -g M=1
        declare -g B=1
        declare -g R=1
        declare -g P=1
        declare -g G=10000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 8 ]]; then
        echo "================== 8.3B LLAMA2 (1 node)"
        declare -g m=8.3
        declare -g H=3072
        declare -g F=11008
        declare -g N=72
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 10 ]]; then
        echo "================== 10B LLAMA2 (1 node)"
        declare -g m=10
        declare -g H=4096
        declare -g F=12400
        declare -g N=50
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 13 ]]; then
        echo "================== 13B LLAMA2 (1 node)"
        declare -g m=13
        declare -g H=5120
        declare -g F=13824
        declare -g N=40
        declare -g L=40
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 20 ]]; then
        echo "================== 20B ZeRO paper (1 node)"
        declare -g m=20
        declare -g H=5120
        declare -g F=20480
        declare -g N=40
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 30 ]]; then
        echo "================== 30B LLAMA2 (1 node)"
        declare -g m=30
        declare -g H=6656
        declare -g F=17920
        declare -g N=60
        declare -g L=52
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=4
        declare -g A=0
        declare -g I=1
    elif [[ $model_size == 40 ]]; then
        echo "================== 40B GPT-2 (1 node)"
        declare -g m=40
        declare -g H=5120
        declare -g F=20480
        declare -g N=128
        declare -g L=40
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=8
        declare -g A=0
        declare -g I=1
    elif [[ $model_size == 70 ]]; then
        echo "================== 70B LLAMA2 (1 nodes)"
        declare -g m=70
        declare -g H=8192
        declare -g F=28672
        declare -g N=80
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=5
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=1
        declare -g G=100000000
        declare -g D=8
        declare -g A=0
        declare -g I=1
    # ================= TWINFLOW EXPTS END =================
    else
        echo "NNODES not in defined list  (NNODES = $NNODES)"
        exit 1
    fi
    }
# m:H:F:N:L:U:S:K:T:M:B:R:G:P:D:A:O:

############### Run for diff model sizes.
model_sizes=(3)
for model_size in "${model_sizes[@]}"; do
    set_model_size $model_size
    B=$((M * D ))
    bash config-n-run.sh -m $model_size -H $H -F $F -N $N -L $L -U $U -S $S -K $K -M $M -B $B -I $I -P $P -T $T -D $D
done
############### Run for diff model sizes.



