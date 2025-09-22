## Evaluating LLM restart performance
The idea of this project is to thoroughly characterize different I/O bottlenecks while restarting an LLM training from a previously captured checkpoint. 
We start with Microsoft's DeepSpeed training runtime, and evaluate the restart performance of different parallelism strategies (data, pipeline, tensor) using ZeRO-1 (i.e., shard the optimizer state across all data-parallel replicas). Initially, the goal is to capture the number of files, size of each file, and time taken to checkpoint, and the time taken to restart, for different parallelism strategies for small (3B), medium (13B), and large model (70B).

The submit-job.sh is the entry bash file through which models of different sizes can be selected, and be passed on to the config-n-run file, which in turn converts these args into deepspeed specific config parameters and command line instructions, finally executing the model.

#### Installing TorchSnapshot Checkpointing Engine
The default version of TorchSnapshot doesn't work well with DeepSpeed because it assumes each rank as a data-parallel rank and assumes redundancy. However, since DeepSpeed would have distinct elements on each rank, we disable this inside torchsnapshot/pg_wrapper.py. The modified source code is placed in this repo.
```
cd torchsnapshot
python setup.py install
```
For additional installation/debugging instructions, please refer: https://github.com/pytorch/torchsnapshot.

#### Installing DataStates-LLM Checkpointing Engine
This is based on HPDC'24 paper: "Datastates-llm: lazy asynchronous checkpointing for large language models". This can be installed using
```
git clone git@github.com:DataStates/datastates-llm.git
cd datastates-llm
git checkout revamp
bash install.sh
```

#### Testing with different checkpointing engines in DeepSpeed
```
git clone git@github.com:DataStates/DeepSpeed.git
cd DeepSpeed
git checkout multi-ckpt-engines
pip install .
```
Please refer to https://www.deepspeed.ai/tutorials/advanced-install/ page for more details.

