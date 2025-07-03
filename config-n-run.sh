#!/bin/bash



restart_perf_env
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

while getopts ":m:H:F:N:L:U:S:K:M:B:P:T:I:D:" opt; do
  case $opt in
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
HOSTFILE="$output_dir/hostfile"
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
sort -u $PBS_NODEFILE > $HOSTFILE
#sed "s/$/ slots=$NRANKS_PER_NODE/" $PBS_NODEFILE > $HOSTFILE
NPROCS=$((NRANKS_PER_NODE * NNODES))
MASTER_ADDR=$(head -n1 $PBS_NODEFILE)
MASTER_PORT=6000
DARSHAN_LIB=/soft/perftools/darshan/darshan-3.4.4/lib/libdarshan.so
export LD_PRELOAD=$DARSHAN_LIB
export DARSHAN_LOG_DIR_BASE=$HOME/restart_perf/darshan_logs
mkdir -p $DARSHAN_LOG_DIR_BASE
MPI_LAUNCH_PARAMS="mpirun -np $NPROCS -hostfile $HOSTFILE \
	-env MASTER_ADDR=$MASTER_ADDR \
	-env MASTER_PORT=$MASTER_PORT
	-env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
	-env PATH="$PATH" \
	-env CUDA_VISIBLE_DEVICES="0,1,2,3" \
	-env LD_PRELOAD="$LD_PRELOAD" \
	"

DEEPSPEED_PARAMS="python -u -m deepspeed.launcher.launch \
	--no_local_rank 
	"
NCCL_IB_DISABLE=1 >> .deepspeed_env
NCCL_SOCKET_IFNAME=eth0 >> .deepspeed_env

sed -i 's/,//g' $HOSTFILE        # Remove commas
sed -i 's/[[:space:]]*$//' $HOSTFILE  # Remove trailing spaces
LAUNCH_PARAMS="--hostfile=$HOSTFILE"
#

USE_DEEPSPEED=1
ZERO_STAGE=1

EXIT_INTERVAL=20
DP=$(((NNODES * 4) / (PP * TP)))
WORLD_SIZE=$((TP*PP*DP))

CHECKPOINT_PATH=/grand/VeloC/mikailg/DeepSpeed-restart-perf/tp${TP}_pp${PP}_dp${DP} 
mkdir -p $CHECKPOINT_PATH

LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1
EXP_DIR=${HOME}/DeepSpeed-restart_perf-logs
mkdir -p $EXP_DIR
LOG_DIR="${EXP_DIR}/tp${TP}_pp${PP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_cont"
mkdir -p $LOG_DIR

# --ffn-hidden-size $FFN_HIDDEN_SIZE \
options=" \
	--tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --ffn-hidden-size $FFN_HIDDEN_SIZE
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --save $CHECKPOINT_PATH \
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
        --no-pipeline-parallel \
        "
# --cpu-optimizer \
# --use-rotary-position-embeddings \

cat <<EOT > $CONFIG_JSON
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
		"grad_accum_dtype": "bf16"
 	},
	"wall_clock_breakdown": false,
	"memory_breakdown": false,
	"flops_profiler": {
		"enabled": false
	}
}
EOT

log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH"
rm -rf $output_dir/log-$log_str.log
# pdsh -w "$(awk '{printf "%s%s",sep,$1; sep=","}' $COBALT_NODEFILE)" 'rm -rf /local/scratch/*'
# eval "rm -rf $CHECKPOINT_PATH"
TMPDIR=/local/scratch/nsys-profile
mpirun -n $NNODES mkdir -p $TMPDIR
run_cmd="{ time deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} ;} | tee -a $output_dir/log-$log_str.log"
#run_cmd="nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys -t cuda,nvtx,osrt --mpi-impl=mpich ${MPI_LAUNCH_PARAMS} deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1"
run_cmd="${MPI_LAUNCH_PARAMS} nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys -t cuda,nvtx,osrt --trace-fork-before-exec true --mpi-impl=mpich deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1"
echo $run_cmd

# echo ${run_cmd}
eval ${run_cmd}
ls -ltrh "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
# eval "rm -rf $CHECKPOINT_PATH"
# rm -rf /local/scratch/*
# set +x
