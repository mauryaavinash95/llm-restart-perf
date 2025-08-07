#!/bin/bash

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
iter=0
EXIT_ITER=0


while getopts ":i:m:H:F:N:L:U:S:K:M:B:P:T:I:D:E:" opt; do
  case $opt in
    i)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        iter="$OPTARG"
      else
        echo "Invalid iteration number: $OPTARG is not a valid integer." >&2
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
    E)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        EXIT_ITER="$OPTARG"
      else
        unset EXIT_ITER
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

if [[ "$model_size_B" -eq 3 && "$TP" -eq 1 ]]; then
    echo "Invalid configuration, skipping"
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
echo "EXIT ITER: $EXIT_ITER"

DIR=$HOME/restart_perf/Megatron-DeepSpeed/
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
BASE_DATA_PATH=$HOME/restart_perf/dataset
DATASET="${BASE_DATA_PATH}/my-gpt2_text_document"
TOKENIZER_PATH=$HOME/restart_perf/dataset/tokenizer.model
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

output_dir="/grand/VeloC/mikailg/DeepSpeed-restart-perf/${model_size_B}Bparams/tp${TP}-iter${iter}/"
mkdir -p "$output_dir"
CONFIG_JSON="$output_dir/ds_config.json"
DEEPSPEED_HOSTFILE="$output_dir/ds_hostfile"

echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "Number of nodes found as $NNODES"


USE_DEEPSPEED=1
ZERO_STAGE=1

EXIT_INTERVAL=20
#DP=$(((NNODES * 4) / (PP * TP)))
WORLD_SIZE=$((TP*PP*DP))


LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1
CHECKPOINT_PATH=/grand/VeloC/mikailg/DeepSpeed-restart-perf/modelsize${model_size_B}_tp${TP}_pp${PP}_dp${DP}
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH



DARSHAN_LIB=$HOME/restart_perf/software/installs/darshan/lib/libdarshan.so
DARSHAN_LOGDIR=$CHECKPOINT_PATH/darshan_logs
DARSHAN_CONFIG_PATH=$HOME/restart_perf/llm-restart-perf/darshan_config.cfg
mkdir -p $DARSHAN_LOGDIR
# export LD_PRELOAD=$DARSHAN_LIB:$LD_PRELOAD

echo "NCCL_IB_DISABLE=1" >> .deepspeed_env
echo "NCCL_SOCKET_IFNAME=eth0" >> .deepspeed_env
echo "DARSHAN_LOGDIR=$DARSHAN_LOGDIR" >> .deepspeed_env
echo "DARSHAN_ENABLE_NONMPI=1" >> .deepspeed_env
echo "DARSHAN_CONFIG_PATH=$DARSHAN_CONFIG_PATH" >> .deepspeed_env
echo "DARSHAN_MMAP_LOG_PATH=$DARSHAN_MMAP_LOG_PATH" >> .deepspeed_env
echo "LD_PRELOAD=$DARSHAN_LIB" >> .deepspeed_env
echo "EXIT_AFTER_IT=$EXIT_ITER" >> .deepspeed_env

# --ffn-hidden-size $FFN_HIDDEN_SIZE \
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
		"grad_accum_dtype": "fp32"
 	},
	"wall_clock_breakdown": false,
	"memory_breakdown": false,
	"flops_profiler": {
		"enabled": false
	}
}
EOT

log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH-iter$iter"
#TMPDIR=/local/scratch/nsys-profile
#mpirun -n $NNODES mkdir -p $TMPDIR
#export NSYS_LAUNCH_LOG=debug
if [ "$TP" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$TP" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$TP" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported T value: $TP"
    exit 1
fi

export EXIT_AFTER_IT=1
vmtouch -e $CHECKPOINT_PATH
run_cmd="nsys profile --force-overwrite true -o $TMPDIR/log-$log_str-nsys-checkpoint -t cuda,nvtx -e LD_PRELOAD=$DARSHAN_LIB:$LD_PRELOAD,DARSHAN_LOGDIR=$DARSHAN_LOGDIR,DARSHAN_ENABLE_NONMPI=1,DARSHAN_CONFIG_PATH=$DARSHAN_CONFIG_PATH deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str-before-stop.log 2>&1"
#run_cmd="deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str-before-stop.log 2>&1"
echo $run_cmd
eval ${run_cmd}
mkdir -p $output_dir/checkpoint
mv $TMPDIR/*.nsys-rep $output_dir/checkpoint
mv /tmp/mgossman_python*.darshan $output_dir/checkpoint
rm -rf /tmp/mgossman_*

vmtouch -e $CHECKPOINT_PATH

if [ $((EXIT_AFTER_IT)) -gt 0 ]; then 
  vmtouch -e $CHECKPOINT_PATH
  unset EXIT_AFTER_IT
  run_cmd="nsys profile --force-overwrite true -o $TMPDIR/log-$log_str-nsys-restart -t cuda,nvtx -e LD_PRELOAD=$DARSHAN_LIB:$LD_PRELOAD,DARSHAN_LOGDIR=$DARSHAN_LOGDIR,DARSHAN_ENABLE_NONMPI=1,DARSHAN_CONFIG_PATH=$DARSHAN_CONFIG_PATH deepspeed ${DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str-after-stop.log 2>&1"
  eval ${run_cmd}
  mkdir -p $output_dir/restart
  mv /tmp/mgossman_python* $output_dir/restart
  mv $TMPDIR/*.nsys-rep $output_dir/restart
fi

tar -czvf ${log_str}.tar.gz $output_dir
ls -ltrh "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
