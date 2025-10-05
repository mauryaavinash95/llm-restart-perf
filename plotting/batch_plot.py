import json
import re
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_throughput(results):
    throughput = {}
    for strategy, sizes in results.items():
        throughput[strategy] = {}
        for size_str, size_data in sizes.items():
            size_MB = int(size_str)
            rank_results = size_data["rank_results"]

            all_durations = []
            for rank_result in rank_results:
                trial_times = list(rank_result[size_str][strategy])
                all_durations.extend(trial_times)

            total_bytes = size_MB * 1024 * 1024 * len(rank_results)
            max_time = max(all_durations)
            agg_throughput = total_bytes / max_time / (1024 ** 3)  # GB/s
            throughput[strategy][size_MB] = agg_throughput
    return throughput

def parse_nodes_ranks(filename):
    m = re.search(r"results_(\d+)_(\d+)\.json", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def plot_dir(results_dir, out="throughput.png"):
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("results_*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    line_styles = {1: "-", 2: "--", 4: "-.", 8: ":"}
    markers = {1: "o", 2: "s", 4: "^"}

    plt.figure(figsize=(10, 6))

    # Collect all sizes globally for consistent x-axis
    all_sizes = set()
    for f in json_files:
        results = load_results(f)
        for strat in results.values():
            all_sizes.update(int(sz) for sz in strat.keys())
    sizes = sorted(all_sizes)
    x_labels = [str(s) for s in sizes]
    x = range(len(sizes))

    for f in json_files:
        nodes, ranks = parse_nodes_ranks(f.name)
        if nodes is None or ranks is None:
            print(f"Skipping {f.name}, cannot parse nodes/ranks")
            continue

        results = load_results(f)
        throughput = compute_throughput(results)

        for strategy, size_map in throughput.items():
            y = [size_map.get(sz, None) for sz in sizes]
            style = line_styles.get(nodes, "-")
            marker = markers.get(ranks, "o")
            label = f"{strategy} ({nodes} nodes, {ranks} ranks/node)"
            plt.plot(x, y, linestyle=style, marker=marker, label=label)

    plt.xticks(x, x_labels, rotation=45)
    plt.xlabel("Problem size (MB per rank)")
    plt.ylabel("Max aggregated throughput (GB/s)")
    plt.title("I/O Throughput by File Aggregation Strategy")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_throughput.py results_dir [out.png]")
        sys.exit(1)

    results_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "batch_throughput.png"
    plot_dir(results_dir, out)
