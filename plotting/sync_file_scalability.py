import json
import sys
import os
import matplotlib.pyplot as plt

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

# def compute_throughput(results):
#     """
#     Compute aggregated throughput per strategy and size.
#     Returns dict: strategy -> { size_MB -> throughput_GBps }
#     """
#     throughput = {}
#     for strategy, sizes in results.items():
#         throughput[strategy] = {}
#         for size_str, size_data in sizes.items():
#             size_MB = int(size_str)
#             rank_results = size_data["rank_results"]

#             # Collect all trial durations
#             all_durations = []
#             for rank_result in rank_results:
#                 trial_times = list(rank_result[size_str][strategy])
#                 all_durations.extend(trial_times)

#             # Throughput = total_bytes / max_time (GB/s)
#             total_bytes = size_MB * 1024 * 1024 * len(rank_results)
#             max_time = max(all_durations)
#             agg_throughput = total_bytes / max_time / (1024 ** 3)  # GB/s
#             throughput[strategy][size_MB] = agg_throughput
#     return throughput

def compute_throughput(results):
    """
    Compute average max-aggregated throughput per strategy and size.
    For each trial, throughput is total_bytes / slowest rank time.
    Then average across trials.
    Returns dict: strategy -> { size_MB -> avg_max_aggregated_throughput_GBps }
    """
    throughput = {}
    for strategy, sizes in results.items():
        throughput[strategy] = {}
        for size_str, size_data in sizes.items():
            size_MB = int(size_str)
            rank_results = size_data["rank_results"]

            # Determine number of trials
            num_trials = len(rank_results[0][size_str][strategy])
            trial_throughputs = []

            for t in range(num_trials):
                # For this trial, get each rank's duration
                rank_times = [rank_result[size_str][strategy][t] for rank_result in rank_results]
                max_time = max(rank_times)  # slowest rank dominates
                total_bytes = size_MB * 1024 * 1024 * len(rank_results)  # sum across ranks
                trial_tp = total_bytes / max_time / (1024 ** 3)  # GB/s
                trial_throughputs.append(trial_tp)

            # Average across trials
            avg_tp = sum(trial_throughputs) / len(trial_throughputs)
            throughput[strategy][size_MB] = avg_tp

    return throughput



def plot_throughput(throughput, results_file, out=None):
    # Use results filename to generate plot filename and title if not provided
    base_name = os.path.basename(results_file).replace(".json", "")
    if out is None:
        out = f"{base_name}.png"
    
    # Collect all problem sizes across strategies
    sizes = sorted({s for strat in throughput.values() for s in strat.keys()})
    x_labels = [str(s) for s in sizes]
    x = range(len(sizes))  # categorical positions

    plt.figure(figsize=(10, 6))
    for strategy, size_map in throughput.items():
        y = [size_map.get(sz, None) for sz in sizes]
        plt.plot(x, y, marker="o", label=strategy)

    # Format x-axis as categorical
    plt.xticks(x, x_labels, rotation=45)
    plt.xlabel("Problem size (MB per rank)")
    plt.ylabel("Max aggregated throughput (GB/s)")
    plt.title(f"I/O Throughput for {base_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_throughput.py results.json [out.png]")
        sys.exit(1)

    results_file = sys.argv[1]
    results = load_results(results_file)
    throughput = compute_throughput(results)
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    plot_throughput(throughput, results_file, out_file)
