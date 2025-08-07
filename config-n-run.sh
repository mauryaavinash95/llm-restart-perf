#!/bin/bash

source ~/.bash_profile
dlconda
# Taken from: https://docs.alcf.anl.gov/polaris/applications-and-libraries/libraries/nccl/
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_REQ_BUF_SIZE=16MB
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
export FI_CXI_RDZV_THRESHOLD=2000
unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET

CKPT_APPROACH=0
HOST_CACHE=0
model_size_B=0
HIDDEN_SIZE=0
FFN_HIDDEN_SIZE=0
NUM_LAYERS=0
NUM_HEADS=0
SEQ_LENGTH=0
NUM_KV_HEADS=0
TRAIN_ITERS=0
NNODES=$(wc -l < $PBS_NODEFILE)
PP=2
TP=4
DP=1
SAVE_INTERVAL=1
MICRO_BATCH=1
GLOBAL_BATCH=1

while getopts ":c:h:m:H:F:N:L:U:S:K:M:B:P:T:I:D:" opt; do
  case $opt in
    c)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        CKPT_APPROACH="$OPTARG"
      else
        echo "Invalid CKPT_APPROACH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    h)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        HOST_CACHE="$OPTARG"
      else
        echo "Invalid HOST_CACHE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    m)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        model_size_B="$OPTARG"
      else
        echo "Invalid model_size_B: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    H)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    F)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        FFN_HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid FFN_HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    N)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_LAYERS="$OPTARG"
      else
        echo "Invalid NUM_LAYERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    L)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_HEADS="$OPTARG"
      else
        echo "Invalid NUM_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    U)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        SEQ_LENGTH="$OPTARG"
      else
        echo "Invalid SEQ_LENGTH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    S)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_KV_HEADS="$OPTARG"
      else
        echo "Invalid NUM_KV_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    K)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TRAIN_ITERS="$OPTARG"
      else
        echo "Invalid TRAIN_ITERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    M)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        MICRO_BATCH="$OPTARG"
      else
        echo "Invalid MICRO_BATCH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    B)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        GLOBAL_BATCH="$OPTARG"
      else
        echo "Invalid GLOBAL_BATCH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    P)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        PP="$OPTARG"
      else
	 echo "Invalid PIPELINE PARALLELISM: $OPTARG is not a valid integer." >&2
	 exit 1
      fi
      ;;
    T)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TP="$OPTARG"
      else
        echo "Invalid TESNOR PARALLELISM: $OPTARG is not a valid integer." >&2
	exit 1
      fi
      ;;
    I)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        SAVE_INTERVAL="$OPTARG"
      else
        echo "Invalid SAVE_INTERVAL: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    D)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        DP="$OPTARG"
      else
        DP=1
      fi
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$model_size_B" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$FFN_HIDDEN_SIZE" ] || [ -z "$NUM_LAYERS" ] || [ -z "$NUM_HEADS" ] || [ -z "$SEQ_LENGTH" ] || [ -z "$NUM_KV_HEADS" ] || [ -z "$TRAIN_ITERS" ]; then
  echo "Missing required parameter(s)." >&2
  exit 1
fi

# Perform further processing with the parsed parameters
echo "CKPT_APPROACH: $CKPT_APPROACH"
echo "HOST_CACHE: $HOST_CACHE"
echo "model_size_B: $model_size_B"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "FFN_HIDDEN_SIZE: $FFN_HIDDEN_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "NUM_HEADS: $NUM_HEADS"
echo "SEQ_LENGTH: $SEQ_LENGTH"
echo "NUM_KV_HEADS: $NUM_KV_HEADS"
echo "TRAIN_ITERS: $TRAIN_ITERS"
echo "PIPE PARALLEL: $PP"
echo "TENSOR PARALLEL: $TP"
echo "DATA PARALLEL: $DP"
echo "SAVE_INTERVAL: $SAVE_INTERVAL"

if [[ "$model_size_B" -eq 3 && "$TP" -eq 1 ]]; then
    echo "Invalid Configurations, skipping"
    exit 1
fi

DIR=$HOME/restart_perf/Megatron-DeepSpeed/
cd ${DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=$HOME/restart_perf/dataset
DATASET="${BASE_DATA_PATH}/my-gpt2_text_document"
TOKENIZER_PATH=$HOME/restart_perf/dataset/tokenizer.model
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

output_dir="$HOME/restart_perf/outputs/${model_size_B}B-output/llama2-NN$NNODES/"
mkdir -p "$output_dir"
CONFIG_JSON="$output_dir/ds_config.json"
DEEPSPEED_HOSTFILE="$output_dir/ds_hostfile"
MPI_HOSTFILE="$output_dir/mpi_hostfile"
echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
#echo "CC=gcc" >> .deepspeed_env
#echo "CXX=g++" >> .deepspeed_env
#echo "IBV_FORK_SAFE=1" >> .deepspeed_env
#echo "CFLAGS=-I/soft/datascience/conda/2023-01-10/mconda3/include/" >> .deepspeed_env
#echo "LDFLAGS=-L/soft/datascience/conda/2023-01-10/mconda3/lib/" >> .deepspeed_env
#echo "CUDA_DEVICE_MAX_CONNECTIONS=1" >> .deepspeed_env
#echo "TORCHSNAPSHOT_PER_RANK_MEMORY_BUDGET_BYTES=34359738368" >> .deepspeed_env
#echo "_DEFAULT_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env
#echo "_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env


echo "Number of nodes found as $NNODES"
NRANKS_PER_NODE=4
#if need to change to other number of GPUs/host
if [ $((DP*TP)) -gt 3 ]; then
  NRANKS_PER_NODE=4
else
  NRANKS_PER_NODE=$((DP*TP))
fi
WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
LAUNCH_PARAMS="--include localhost: "
for ((gpu_id=0; gpu_id<NRANKS_PER_NODE; gpu_id++)); do
    LAUNCH_PARAMS+="$gpu_id"
    if [ $gpu_id -lt $((NRANKS_PER_NODE - 1)) ]; then
        LAUNCH_PARAMS+=","
    fi
done
sort -u $PBS_NODEFILE > $MPI_HOSTFILE
sort -u $PBS_NODEFILE > $DEEPSPEED_HOSTFILE
sed "s/$/ slots=$NRANKS_PER_NODE/" $PBS_NODEFILE > $DEEPSPEED_HOSTFILE
NPROCS=$((NRANKS_PER_NODE * NNODES))
MASTER_ADDR=$(head -n1 $PBS_NODEFILE)
MASTER_PORT=6006

export MPICH_GPU_SUPPORT_ENABLED=1
CHECKPOINT_PATH=/grand/VeloC/mikailg/DeepSpeed-restart-perf/modelsize${model_size_B}_tp${TP}_pp${PP}_dp${DP}
mkdir -p $CHECKPOINT_PATH
DARSHAN_LIB=$HOME/restart_perf/software/installs/darshan/lib/libdarshan.so
export DARSHAN_LOGS=$CHECKPOINT_PATH/darshan_logs
#export LD_PRELOAD=$DARSHAN_LIB
#export DARSHAN_ENABLE_NONMPI=1
mkdir -p $DARSHAN_LOGS/$(date +%Y/%-m/%-d)
MPI_LAUNCH_PARAMS="mpirun -np $NPROCS --hostfile $MPI_HOSTFILE \
	-env MASTER_ADDR=$MASTER_ADDR \
	-env MASTER_PORT=$MASTER_PORT \
	-env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
	-env PATH="$PATH" \
	-env LD_PRELOAD=${DARSHAN_LIB} \
	-env DARSHAN_LOGDIR=${DARSHAN_LOGS} \
	-env DARSHAN_ENABLE_NONMPI=1 \
	-env MPICH_GPU_SUPPORT_ENABLED=1 \
	"

DEEPSPEED_PARAMS="python -u -m deepspeed.launcher.launch \
	--no_local_rank 
	"
NCCL_IB_DISABLE=1 >> .deepspeed_env
NCCL_SOCKET_IFNAME=eth0 >> .deepspeed_env

sed -i 's/,//g' $DEEPSPEED_HOSTFILE        # Remove commas
sed -i 's/[[:space:]]*$//' $DEEPSPEED_HOSTFILE  # Remove trailing spaces
LAUNCH_PARAMS="--hostfile=$DEEPSPEED_HOSTFILE"
#

USE_DEEPSPEED=1
ZERO_STAGE=1

EXIT_INTERVAL=20
DP=$(((NNODES * 4) / (PP * TP)))
WORLD_SIZE=$((TP*PP*DP))


LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1

options=" \
	--tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATASET \
       --vocab-file ${VOCAB_PATH} \
	     --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval $SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 0 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads ${NUM_KV_HEADS} \
       --deepspeed \
       --exit-interval ${EXIT_INTERVAL} \
       --deepspeed_config=${CONFIG_JSON} \
       --zero-stage=${ZERO_STAGE} \
        --checkpoint-activations \
        --deepspeed-activation-checkpointing \
        "

# Compose common config
COMMON_CONFIG=$(cat <<EOC
{
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "bf16": {
        "enabled": true
    },
    "data_types": {
        "grad_accum_dtype": "fp32"
    },
    "wall_clock_breakdown": true,
    "memory_breakdown": false,
    "flops_profiler": {
        "enabled": true
    }
EOC
)

# Decide on checkpoint approach and extension
case $CKPT_APPROACH in
    0)
        echo "Checkpointing using None Checkpointing approach"
        CKPT_STANZA=', "none_ckpt_config": true'
        ;;
    1)
        echo "Checkpointing with FastPersist approach"
        CKPT_STANZA=', "checkpoint": { "writer": { "type": "FAST", "decoupled": true } }'
        ;;
    2)
        echo "Checkpointing with default Torch.save()"
        CKPT_STANZA=''
        ;;
    3)
        echo "Checkpointing using Python Based AysncTorch approach"
        CKPT_STANZA=', "async_ckpt_config": { "host_cache": -1 }'
        ;;
    4)
        echo "Checkpointing using VELOC Checkpointing approach"
        CKPT_STANZA=', "datastates_ckpt": { "host_cache_size": '"$HOST_CACHE"', "parser_threads": 8 }'
        ;;
    5)
        echo "Checkpointing using TorchSnapshot Async approach"
        CKPT_STANZA=', "torchsnapshot_ckpt": { "enabled": true }'
        ;;
    *)
        echo "Invalid CKPT_APPROACH: $CKPT_APPROACH"
        exit 1
        ;;
esac

# Write full JSON
echo "${COMMON_CONFIG}${CKPT_STANZA}"'}' > "$CONFIG_JSON"


log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH-ckpt$CKPT_APPROACH"
rm -rf $output_dir/log-$log_str.log
# pdsh -w "$(awk '{printf "%s%s",sep,$1; sep=","}' $COBALT_NODEFILE)" 'rm -rf /local/scratch/*'
# eval "rm -rf $CHECKPOINT_PATH"
TMPDIR=/local/scratch/nsys-profile
mpirun -n $NNODES mkdir -p $TMPDIR
export NSYS_LAUNCH_LOG=debug
run_cmd="{ time deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} ;} | tee -a $output_dir/log-$log_str.log"
#run_cmd="nsys profile --force-overwrite true --stop-on-exit true -o $output_dir/log-$log_str-nsys -t cuda,nvtx deepspeed ${DIR}/pretrain_gpt.py ${options} > $output_dir/log-$log_str.log 2>&1"
#run_cmd="${MPI_LAUNCH_PARAMS} nsys profile --force-overwrite=true -o $output_dir/log-$log_str-%p-nsys -t cuda,nvtx,cudnn bash -c 'export PATH=$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH; export LD_PRELOAD=${DARSHAN_LIB}; export DARSHAN_LOGDIR=${DARSHAN_LOGS}; export DARSHAN_ENABLE_NONMPI=1; deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1'"
#run_cmd="export LD_PRELOAD=${DARSHAN_LIB}; export DARSHAN_LOGDIR=${DARSHAN_LOGS}; export DARSHAN_ENABLE_NONMPI=1; deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1"
#echo "deepspeed --no_python ./nsys_test.sh ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options}"
#deepspeed --no_python /home/mgossman/restart_perf/llm-restart-perf/nsys-test.sh ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options}
if [ "$TP" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$TP" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$TP" -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2
elif [ "$TP" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported T value: $TP"
    exit 1
fi
#export LD_PRELOAD=${DARSHAN_LIB}; export DARSHAN_LOGDIR=${DARSHAN_LOGS}; export DARSHAN_ENABLE_NONMPI=1;
run_cmd="nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys-before-stop -t cuda,nvtx,osrt deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str-before-stop.log 2>&1"
echo $run_cmd

export EXIT_AFTER_5=1
eval ${run_cmd}
unset EXIT_AFTER_5
run_cmd="nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys-after-stop -t cuda,nvtx,osrt deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str-after-stop.log 2>&1"
eval ${run_cmd}

ls -ltrh "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
# eval "rm -rf $CHECKPOINT_PATH"
# rm -rf /local/scratch/*
# set +x
