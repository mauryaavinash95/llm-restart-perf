#!/bin/bash -l
# PBS -l nodes=2:system=polaris
# PBS -l walltime=01:00:00
# PBS -q debug-scaling
# PBS -A VeloC
# PBS -l filesystems=home:grand

source ~/.bash_profile
dlconda
rm -rf ~/.deepspeed_env
# Taken from: https://docs.alcf.anl.gov/polaris/applications-and-libraries/libraries/nccl/
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_CROSS_NIC=1
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NET="AWS Libfabric"
# export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_CXI_DEFAULT_TX_SIZE=131072
# export FI_CXI_RDZV_PROTO=alt_read
# export FI_CXI_RX_MATCH_MODE=software
# export FI_CXI_REQ_BUF_SIZE=16MB
# export FI_CXI_RDZV_GET_MIN=0
# export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
# export FI_CXI_RDZV_THRESHOLD=2000
unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET

# The following is recommended by ChatGPT
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
export NCCL_SOCKET_IFNAME=hsn0
# export NCCL_NET=ofi
# export FI_PROVIDER=cxi

# set -x
# Define default values
LLM_HOST="jlse"
if [[ -n "$PBS_NODEFILE" ]]; then
    NODEFILE="$PBS_NODEFILE"
    LLM_HOST="polaris"
elif [[ -n "$COBALT_NODEFILE" ]]; then
    NODEFILE="$COBALT_NODEFILE"
    LLM_HOST="jlse"
else
    echo "Error: Neither PBS_NODEFILE nor COBALT_NODEFILE is set."
    exit 1
fi

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
NNODES=$(wc -l < $NODEFILE)
PP=$NNODES
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
        FFN_HIDDEN_SIZE=0
        # echo "Invalid FFN_HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
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
        PP=$NNODES
      fi
      ;;
    T)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TP="$OPTARG"
      else
        TP=4
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
if [ -z "$CKPT_APPROACH" ] || [ -z "$HOST_CACHE" ] || [ -z "$model_size_B" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$FFN_HIDDEN_SIZE" ] || [ -z "$NUM_LAYERS" ] || [ -z "$NUM_HEADS" ] || [ -z "$SEQ_LENGTH" ] || [ -z "$NUM_KV_HEADS" ] || [ -z "$TRAIN_ITERS" ]; then
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

DIR=$HOME/dl-io/Megatron-DeepSpeed/
echo "PATH=${PATH}:/soft/applications/conda/2024-10-30-workshop/mconda3/include/" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/soft/applications/conda/2024-10-30-workshop/mconda3/lib/" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=gcc" >> .deepspeed_env
echo "CXX=g++" >> .deepspeed_env
echo "IBV_FORK_SAFE=1" >> .deepspeed_env
echo "CFLAGS=-I/soft/applications/conda/2024-10-30-workshop/mconda3/include/" >> .deepspeed_env
echo "LDFLAGS=-L/soft/applications/conda/2024-10-30-workshop/mconda3/lib/" >> .deepspeed_env
echo "CPATH=$CPATH:$HOME/softwares/liburing/include" >> .deepspeed_env
echo "LIBRARY_PATH=$LIBRARY_PATH:$HOME/softwares/liburing/lib" >> .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/softwares/liburing/lib" >> .deepspeed_env
echo "TORCHSNAPSHOT_PER_RANK_MEMORY_BUDGET_BYTES=21474836480" >> .deepspeed_env # 20 GB
echo "_DEFAULT_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env
echo "TORCHSNAPSHOT_MAX_PER_RANK_IO_CONCURRENCY_OVERRIDE=1" >> .deepspeed_env
echo "TORCHSNAPSHOT_DISABLE_BATCHING=1" >> .deepspeed_env
echo "_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=$HOME/dl-io/datasets
DATASET="${BASE_DATA_PATH}/meg-gpt2_text_document"
TOKENIZER_PATH=$HOME/dl-io/datasets/llama2/tokenizer.model
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

output_dir="$HOME/dl-io/vlcc-datastates/outputs-${LLM_HOST}/${model_size_B}B"
mkdir -p "$output_dir"
CONFIG_JSON="$output_dir/ds_config.json"
HOSTFILE="$output_dir/hostfile"

echo "Number of nodes found as $NNODES"
NRANKS_PER_NODE=4
sed "s/$/ slots=$NRANKS_PER_NODE/" $NODEFILE > $HOSTFILE
LAUNCH_PARAMS="--hostfile=$HOSTFILE"

USE_DEEPSPEED=1
ZERO_STAGE=1

EXIT_INTERVAL=2000
DP=$(((NNODES * 4) / (PP * TP)))
WORLD_SIZE=$((TP*PP*DP))

if [[ "$LLM_HOST" == "polaris" ]]; then
  CHECKPOINT_PATH=/eagle/projects/argonne_tpc/am6429/scratch/llama2/tp${TP}_pp${PP}_dp${DP} #_saveint${SAVE_INTERVAL}
  # CHECKPOINT_PATH=/local/scratch/llama2/tp${TP}_pp${PP}_dp${DP} 
else
  CHECKPOINT_PATH=/vast/users/amaurya/scratch/llama2/tp${TP}_pp${PP}_dp${DP}
fi
LOAD_CHECKPOINT_PATH=$CHECKPOINT_PATH

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
       --data-path $DATASET \
       --vocab-file ${VOCAB_PATH} \
	     --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
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

# --no-pipeline-parallel \
# --cpu-optimizer \
# --use-rotary-position-embeddings \

# AM comment: BP16 does not work with deepspeed for now
# https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options
# So switching to regular FP16.
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
    "memory_breakdown": true,
    "flops_profiler": {
        "enabled": false
    }
EOC
)

# Decide on checkpoint approach and extension
case $CKPT_APPROACH in
    0)
        echo "Checkpointing using None Checkpointing approach"
        CKPT_STANZA=', "none_ckpt": true'
        ;;
    1)
        echo "Checkpointing with default Torch.save()"
        CKPT_STANZA=''
        ;;
    2)
        echo "Checkpointing with FastPersist approach"
        CKPT_STANZA=', "checkpoint": { "writer": { "type": "FAST", "decoupled": true } }'
        ;;
    3)
        echo "Checkpointing using DataStates Base Checkpointing approach"
        CKPT_STANZA=', "datastates_ckpt": { "host_cache_size": '"$HOST_CACHE"', "engine_type": "simple_engine", "profile_engine": true }'
        ;;
    4)
        echo "Checkpointing using DataStates+VLCC Checkpointing approach"
        CKPT_STANZA=', "datastates_ckpt": { "host_cache_size": '"$HOST_CACHE"', "engine_type": "state_engine", "profile_engine": true }'
        ;;
    5)
        echo "Checkpointing using TorchSnapshot Async approach"
        CKPT_STANZA=', "torchsnapshot_ckpt": { "enabled": true }'
        ;;
    6)
        io_buffer_size=4194304
        echo "Checkpointing using FastPersist approach"
        CKPT_STANZA=', "checkpoint": { "writer": { "type": "fast", "decoupled": true, "io_buffer_size": '"$io_buffer_size"', "show_statistics": false, "data_parallel": "replica" } }'
        ;;
    7)
        echo "Checkpointing using DataStates+VLCC+Aggregated Checkpointing approach"
        CKPT_STANZA=', "datastates_ckpt": { "host_cache_size": '"$HOST_CACHE"', "engine_type": "state_aggregated_engine", "profile_engine": true }'
        ;;
    *)
        echo "Invalid CKPT_APPROACH: $CKPT_APPROACH"
        exit 1
        ;;
esac

# Write full JSON
echo "${COMMON_CONFIG}${CKPT_STANZA}"'}' > "$CONFIG_JSON"

eval "rm -rf $HOME/dl-io/Megatron-DeepSpeed/core.*"
log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-gbs$GLOBAL_BATCH-mbs$MICRO_BATCH-iters${TRAIN_ITERS}-ckpt$CKPT_APPROACH-saveint${SAVE_INTERVAL}"
rm -rf $output_dir/log-$log_str.log
echo "PWD is $(pwd)" >> $output_dir/log-$log_str.log
# echo "NSYS_REPORT_DIR=${output_dir}/rep-${log_str}-%n">> .deepspeed_env
pdsh -w "$(awk '{printf "%s%s",sep,$1; sep=","}' $NODEFILE)" 'rm -rf /local/scratch/*'
eval "rm -rf $CHECKPOINT_PATH"
mkdir -p $CHECKPOINT_PATH
cd $CHECKPOINT_PATH || exit 1
run_cmd="{ time deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} ;} | tee -a $output_dir/log-$log_str.log"
# run_cmd="nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys -t cuda,nvtx deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1"
echo $run_cmd

eval ${run_cmd}
ls -ltrh "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
du -s --apparent-size "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
eval "rm -rf $HOME/dl-io/Megatron-DeepSpeed/core.*"
# set +x
