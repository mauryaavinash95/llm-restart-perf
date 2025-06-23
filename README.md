## Evaluating LLM restart performance
The idea of this project is to thoroughly characterize different I/O bottlenecks while restarting an LLM training from a previously captured checkpoint. 
We start with Microsoft's DeepSpeed training runtime, and evaluate the restart performance of different parallelism strategies (data, pipeline, tensor) using ZeRO-1 (i.e., shard the optimizer state across all data-parallel replicas). Initially, the goal is to capture the number of files, size of each file, and time taken to checkpoint, and the time taken to restart, for different parallelism strategies for small (3B), medium (13B), and large model (70B).

The submit-job.sh is the entry bash file through which models of different sizes can be selected, and be passed on to the config-n-run file, which in turn converts these args into deepspeed specific config parameters and command line instructions, finally executing the model.

