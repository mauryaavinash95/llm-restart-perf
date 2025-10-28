import os
import sys
import time
import json
import torch
import gc
import pandas as pd
from mpi4py import MPI
from torchsnapshot import StateDict, Snapshot


def parse_size_to_bytes(size_str):
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if size_str[-1] in units:
        return int(float(size_str[:-1]) * units[size_str[-1]])
    return int(size_str)


def load_model_sizes(csv_path, model_col):
    df = pd.read_csv(csv_path)
    if model_col not in df.columns:
        raise ValueError(f"Model column '{model_col}' not found in CSV")
    buffer_plan = []
    for _, row in df.iterrows():
        size_str = str(row[0])
        count = int(row[model_col])
        if count > 0:
            buffer_plan.append((parse_size_to_bytes(size_str), count))
    return buffer_plan


def allocate_tensor(size_bytes, device):
    """Allocate a float32 tensor of approximately size_bytes bytes."""
    num_elements = size_bytes // 4  
    return torch.randn(num_elements, dtype=torch.float32, device=device)


def benchmark_snapshot_csv(csv_path, model_col, n_trials, base_dir, save_path, restore_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    device = torch.device("cpu") 
    agg_mode = "snapshot"

    save_results = {agg_mode: {}}
    restore_results = {agg_mode: {}}


    buffer_plan = load_model_sizes(csv_path, model_col)
    total_size_all_ranks = sum(buf_size * count for buf_size, count in buffer_plan)


    per_rank_plan = []
    for buf_size, count in buffer_plan:
        base_files = count // world_size
        remainder = count % world_size
        local_count = base_files + (1 if rank < remainder else 0)
        per_rank_plan.append((buf_size, local_count))

    for trial in range(n_trials):
        tensors = []
        paths = []
        for buf_size, local_count in per_rank_plan:
            for i in range(local_count):
                x = allocate_tensor(buf_size, device)
                tensors.append(x)
                paths.append(os.path.join(base_dir, f"snapshot_{buf_size}_rank{rank}_trial{trial}_buf{i}"))

        # SAVE
        aggregate_save_time = 0
        for idx, x in enumerate(tensors):
            ckpt_dict = {"app_state": StateDict(ckpt={"x": x})}
            t0 = time.time()
            Snapshot.take(path=paths[idx], app_state=ckpt_dict, replicated=[])
            aggregate_save_time += time.time() - t0
            del ckpt_dict
        comm.Barrier()


        if(rank == 0):
            print("starting restore")
        # RESTORE
        aggregate_restore_time = 0
        for idx, x in enumerate(tensors):
            snapshot = Snapshot(path=paths[idx])
            restore_dict = {"app_state": StateDict(ckpt={})}
            t1 = time.time()
            snapshot.restore(app_state=restore_dict)
            aggregate_restore_time += time.time() - t1
            del restore_dict, snapshot
        comm.Barrier()


        size_key = str(total_size_all_ranks)
        save_results[agg_mode].setdefault(size_key, {"rank_results": [{}]})
        restore_results[agg_mode].setdefault(size_key, {"rank_results": [{}]})


        rank_save_dict = save_results[agg_mode][size_key]["rank_results"][0]
        rank_restore_dict = restore_results[agg_mode][size_key]["rank_results"][0]

        rank_save_dict.setdefault(f"rank{rank}", []).append(aggregate_save_time)
        rank_restore_dict.setdefault(f"rank{rank}", []).append(aggregate_restore_time)

        del tensors
        gc.collect()

    all_save_results = comm.gather(save_results, root=0)
    all_restore_results = comm.gather(restore_results, root=0)

    if rank == 0:
        merged_save = {agg_mode: {}}
        merged_restore = {agg_mode: {}}

        for result_list, merged in [
            (all_save_results, merged_save),
            (all_restore_results, merged_restore),
        ]:
            for r in result_list:
                for size_key, data in r[agg_mode].items():
                    merged[agg_mode].setdefault(size_key, {"rank_results": [{}]})
                    merged_dict = merged[agg_mode][size_key]["rank_results"][0]
                    for rank_dict in data["rank_results"]:
                        for k, v in rank_dict.items():
                            merged_dict.setdefault(k, []).extend(v)


        with open(save_path, "w") as f:
            json.dump(merged_save, f, indent=2)
        with open(restore_path, "w") as f:
            json.dump(merged_restore, f, indent=2)
        print(f"\n[Rank 0] Results written to:\n {save_path}\n {restore_path}", flush=True)

    torch.cuda.empty_cache()
    gc.collect()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    base_dir = "/grand/VeloC/mikailg/file_scalability/"
    os.makedirs(base_dir, exist_ok=True)

    save_stats, restore_stats = sys.argv[1], sys.argv[2]
    csv_path = sys.argv[3]
    model_col = sys.argv[4]
    n_trials = 3

    benchmark_snapshot_csv(csv_path, model_col, n_trials, base_dir, save_stats, restore_stats)

    comm.Barrier()
    if rank == 0:
        print("\n=== Snapshot benchmark complete ===")


if __name__ == "__main__":
    main()
    sys.exit(0)
