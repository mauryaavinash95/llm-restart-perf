import json
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

def plot_throughput(throughput, title, out="throughput.png"):
    sizes = sorted({s for strat in throughput.values() for s in strat.keys()})
    x_labels = [str(s) for s in sizes]
    x = range(len(sizes))

    plt.figure(figsize=(10, 6))
    for strategy, size_map in throughput.items():
        y = [size_map.get(sz, None) for sz in sizes]
        plt.plot(x, y, marker="o", label=strategy)

    plt.xticks(x, x_labels, rotation=45)
    plt.xlabel("Problem size (MB per rank)")
    plt.ylabel("Max aggregated throughput (GB/s)")
    plt.title(f"I/O Throughput: {title}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out}")

def plot_directory(results_dir):
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("results_*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    for f in json_files:
        print(f"Processing {f.name}...")
        results = load_results(f)
        throughput = compute_throughput(results)
        # Create output file name based on JSON filename
        out_file = results_dir / f"{f.stem}-throughput.png"
        plot_throughput(throughput, title=f.name, out=str(out_file))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_throughput.py results_dir")
        sys.exit(1)

    results_dir = sys.argv[1]
    plot_directory(results_dir)
