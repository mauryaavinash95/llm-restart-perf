#!/bin/bash -l
#PBS -l nodes=1
#PBS -l walltime=09:00:00
#PBS -q preemptable
#PBS -A VeloC
#PBS -l filesystems=home:grand
#PBS -r y

echo "Submitted job"
NNODES=$(wc -l < $PBS_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"

source ~/.bashrc
restart_perf_env
module load nvhpc-mixed/23.9
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
        declare -g K=5
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
        echo "================== 3B BLOOM model (1 node): https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json"
        declare -g m=3
        declare -g H=3072         # hidden_size
        declare -g F=8192         # intermediate_size (ffn_hidden_size)
        declare -g N=28           # num_hidden_layers
        declare -g L=24           # num_attention_heads
        declare -g U=131072       # max_position_embeddings (seq length, adjust if needed)
        declare -g S=8            # num_key_value_heads
        declare -g K=5
        declare -g T=4
        declare -g M=16
        declare -g B=1
        declare -g R=1
        declare -g P=1
        declare -g G=10000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
	declare -g E=1
    elif [[ $model_size == 7 ]]; then
        echo "================== 7B LLAMA2 (2 nodes): https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json"
        declare -g m=7
        declare -g H=4096         # hidden_size
        declare -g F=14336        # ffn_hidden_size (intermediate_size)
        declare -g N=32           # num_hidden_layers
        declare -g L=32           # num_attention_heads
        declare -g U=32768        # max_position_embeddings
        declare -g S=8            # num_key_value_heads
        declare -g K=10
        declare -g T=4
        declare -g M=16
        declare -g B=1
        declare -g R=1
        declare -g P=2
        declare -g G=10000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 13 ]]; then
        echo "================== 13B LLAMA2 (1 node): https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/blob/main/config.json"
        declare -g m=13
        declare -g H=5120         # hidden_size
        declare -g F=13824        # ffn_hidden_size (intermediate_size)
        declare -g N=40           # num_hidden_layers
        declare -g L=40           # num_attention_heads
        declare -g U=4096         # max_position_embeddings
        declare -g S=40           # num_key_value_heads
        declare -g K=10
        declare -g T=4
        declare -g M=16
        declare -g B=4
        declare -g R=1
        declare -g P=4
        declare -g G=100000000
        declare -g D=1
        declare -g A=1
        declare -g I=1
    elif [[ $model_size == 33 ]]; then
        echo "================== 33B DeepSeek (8 nodes): https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/config.json"
        declare -g m=33
        declare -g H=7168          # hidden_size
        declare -g F=19200         # ffn_hidden_size (intermediate_size)
        declare -g N=62            # num_hidden_layers
        declare -g L=56            # num_attention_heads
        declare -g U=16384         # max_position_embeddings
        declare -g S=8             # num_key_value_heads
        # declare -g H=6656          # hidden_size
        # declare -g F=17920         # ffn_hidden_size (intermediate_size)
        # declare -g N=64            # num_hidden_layers
        # declare -g L=52            # num_attention_heads
        # declare -g U=16384         # max_position_embeddings
        # declare -g S=4             # num_key_value_heads
        declare -g K=10
        declare -g T=4
        declare -g M=16
        declare -g B=4
        declare -g R=1
        declare -g P=8
        declare -g G=100000000
        declare -g D=1
        declare -g A=0
        declare -g I=1
    elif [[ $model_size == 70 ]]; then
        echo "================== 70B LLAMA2 (1 nodes): https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json"
        declare -g m=70
        declare -g H=8192         # hidden_size
        declare -g F=28672        # ffn_hidden_size (intermediate_size)
        declare -g N=80           # num_hidden_layers
        declare -g L=64           # num_attention_heads
        declare -g U=131072       # max_position_embeddings
        declare -g S=8            # num_key_value_heads
        declare -g K=5
        declare -g T=4
        declare -g M=16
        declare -g B=4
        declare -g R=1
        declare -g P=20
        declare -g G=100000000
        declare -g D=1
        declare -g A=0
        declare -g I=1
    # ================= TWINFLOW EXPTS END =================
    else
        echo "NNODES not in defined list  (NNODES = $NNODES)"
        exit 1
    fi
    set -x
    echo "Forcing some common value of here...."
    declare -g K=2
    declare -g M=16
    declare -g U=2048
    if [[ -v NUM_ITERS ]]; then
        declare -g K=$NUM_ITERS
    fi
    set +x
}
# m:H:F:N:L:U:S:K:T:M:B:R:G:P:D:A:O:

############### Run for diff model sizes.
model_sizes=(3 1)
tensor_sizes=(4 2 1)
for it in {0..2}; do
    for model_size in "${model_sizes[@]}"; do
        set_model_size $model_size
        B=$((M * D ))
        for tensor_parallel in "${tensor_sizes[@]}"; do
            if [[ "$model_size" -eq 3 && "$tensor_parallel" -eq 1 ]]; then
                continue
            fi
            bash ~/restart_perf/llm-restart-perf/config-n-run_cleaned.sh -i $it -m $model_size -H $H -F $F -N $N -L $L -U $U -S $S -K $K -M $M -B $B -I $I -P $P -T $tensor_parallel -D $D -E $E
        done
    done
done
############### Run for diff model sizes.



