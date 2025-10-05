import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# -------------------------
# Hardcode your logs here
# -------------------------
# Each test name has a list of 3 log paths

path1="./7B-output/CKPTAPPR6-TP4-ITER-0/"
path2="./7B-output/CKPTAPPR6-TP4-ITER-1/"
path3="./7B-output/CKPTAPPR6-TP4-ITER-2/"
path4="./7B-output/CKPTAPPR0-TP4-ITER-0/"
path5="./7B-output/CKPTAPPR0-TP4-ITER-1/"
path6="./7B-output/CKPTAPPR0-TP4-ITER-2/"
path7="./7B-output/CKPTAPPR4-TP4-ITER-0/"
path8="./7B-output/CKPTAPPR4-TP4-ITER-1/"
path9="./7B-output/CKPTAPPR4-TP4-ITER-2/"

LOGS = {
    "OUR_AGGREGATION": [path1+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter0.log", 
                        path2+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter1.log", 
                        path3+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter2.log"],
    "No CKPT":         [path4+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt0-iter0.log", 
                        path5+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt0-iter1.log", 
                        path6+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt0-iter2.log"],
    "DataStates":      [path7+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter0.log", 
                        path8+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter1.log", 
                        path9+"log-7B-tp4-pp2-dp2-gbs16-mbs-16-ckpt4-iter2.log"],
}

# Regex for <TIMER:name,value>
TIMER_RE = re.compile(r"<TIMER:([^,>]+),([\d.]+)>")

def parse_log(path):
    """Extract timers from a single log file."""
    timers = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            match = TIMER_RE.search(line)
            if match:
                name = match.group(1).strip()
                value = float(match.group(2))
                timers[name].append(value)
    # Average repeated timers in a single log
    return {k: np.mean(v) for k, v in timers.items()}

def aggregate_tests(log_dict):
    """
    log_dict = {test_name: [log1, log2, log3]}
    Returns {test_name: {timer: avg_over_logs}}
    """
    aggregated = {}
    for test, paths in log_dict.items():
        runs = [parse_log(p) for p in paths]
        all_keys = set().union(*runs)
        averaged = {}
        for key in all_keys:
            vals = [r.get(key, np.nan) for r in runs if key in r]
            averaged[key] = np.mean(vals) if vals else np.nan
        aggregated[test] = averaged
    return aggregated

def plot_results(aggregated):
    """
    aggregated = { test_name: {timer: avg_time} }
    """
    timers = sorted({t for results in aggregated.values() for t in results})
    tests = list(aggregated.keys())

    x = np.arange(len(timers))
    width = 0.8 / len(tests)

    fig, ax = plt.subplots(figsize=(10,6))
    for i, test in enumerate(tests):
        values = [aggregated[test].get(t, np.nan) for t in timers]
        bars = ax.bar(x + i*width, values, width, label=test)

        # Annotate bar heights
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):  # avoid nan labels
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # offset above bar
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width*(len(tests)-1)/2)
    ax.set_xticklabels(timers, rotation=45, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("Average TIMER durations across 3 runs")
    ax.legend()
    plt.tight_layout()
    plt.savefig("deepspeed-7B.png", dpi=150)
    plt.show()
    print(f"Saved plot")


if __name__ == "__main__":
    aggregated = aggregate_tests(LOGS)
    for test, timers in aggregated.items():
        print(f"\n== {test} ==")
        for t, v in timers.items():
            print(f"{t}: {v:.3f}s")
    plot_results(aggregated)
