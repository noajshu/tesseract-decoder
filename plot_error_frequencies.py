#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def plot_one_minus_cdf(csv_path, detector_idx, out_path):
    error_counts = defaultdict(int)
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['detector_index']) == detector_idx:
                error_idx = int(row['error_index'])
                count = int(row['count'])
                error_counts[error_idx] += count
    if not error_counts:
        print(f"No entries found for detector index {detector_idx}.")
        return
    sorted_counts = sorted(error_counts.values(), reverse=True)
    total = sum(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    one_minus_cdf = 1.0 - cumulative / total
    x = np.arange(1, len(sorted_counts) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, one_minus_cdf, marker='o', markersize=3, linewidth=1)
    plt.xlabel("Top N Errors Included")
    plt.ylabel("Fraction of Calls Not Yet Accounted For")
    plt.title(f"1 - CDF of Error Frequency for Detector {detector_idx}")
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved 1-CDF plot to {out_path}")

def plot_detcost_hist(csv_path, detector_idx, out_path):
    detcost_counts = defaultdict(int)
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['detector_index']) == detector_idx:
                val = float(row['detcost'])
                count = int(row['count'])
                detcost_counts[val] += count
    if not detcost_counts:
        print(f"No detcost entries found for detector index {detector_idx}.")
        return
    vals, counts = zip(*sorted(detcost_counts.items()))
    plt.figure(figsize=(8,5))
    plt.bar(vals, counts, width=0.01)
    plt.xlabel("Detcost Value")
    plt.ylabel("Count")
    plt.title(f"Detcost Histogram for Detector {detector_idx}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved detcost histogram to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 1-CDF of error frequencies and detcost histograms for a detector.")
    parser.add_argument("--csv", help="Path to error frequency stats.csv")
    parser.add_argument("--detcost-csv", help="Path to detcost value stats.csv")
    parser.add_argument("--detector", type=int, required=True, help="Detector index to plot")
    parser.add_argument("--out", help="Output image file for error CDF (e.g., cdf_plot.png)")
    parser.add_argument("--hist-out", help="Output image file for detcost histogram")
    args = parser.parse_args()
    if args.csv and args.out:
        plot_one_minus_cdf(args.csv, args.detector, args.out)
    if args.detcost_csv and args.hist_out:
        plot_detcost_hist(args.detcost_csv, args.detector, args.hist_out)
