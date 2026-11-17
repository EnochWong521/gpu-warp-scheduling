"""Validate SM7_QV100 per-launch instruction pairs against the stock-GTO probe."""
import csv
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

from collect_gto_bypass import capture, extract

RESULTS = Path(__file__).resolve().parents[1] / "results/adaptive_policy"
CLASSES = ("arithmetic", "load", "store", "other", "unclassified")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def collect_pairs(run):
    controls = read_csv(RESULTS / "nn_hotspot_limit4.csv")
    archive = read_csv(RESULTS / "gto_bypass.csv")
    rows = []
    for workload, total in [("nn", 32996), ("hotspot", 136822)]:
        stock = [r for r in controls if (r["workload"], r["policy"]) == (workload, "gto")]
        reference = [r for r in archive if r["workload"] == workload]
        validated = list(extract(run, workload, stock))
        blocks = re.split(r"(?m)^kernel_name[ \t]*=[ \t]*", (run / f"{workload}.log").read_text())[1:]
        previous = Counter()
        for block, launch, ref in zip(blocks, validated, reference, strict=True):
            if any(str(value) != ref[key] for key, value in launch.items()):
                raise ValueError(f"Stock bypass diagnostic changed at {workload}/{ref['launch']}")
            entries = capture(r"^gto_probe_pairs =(.*)$", block).split()
            current = Counter()
            for entry in entries:
                pair, count = entry.split(":")
                a, b = pair.split("/")
                if a not in CLASSES or b not in CLASSES or (a, b) in current:
                    raise ValueError(f"Invalid pair {entry}")
                current[a, b] = int(count)
            if previous - current:
                raise ValueError("Cumulative instruction pair count decreased")
            counts = current - previous
            previous = current
            repeats, bypasses = launch["repeats"], launch["bypasses"]
            if sum(counts.values()) != bypasses or launch["unknown_cycles"]:
                raise ValueError(f"Incomplete pair coverage at {workload}/{ref['launch']}")
            for a, b in itertools.product(CLASSES, repeat=2):
                count = counts[a, b]
                rows.append(dict(workload=workload, launch=launch["launch"], greedy=a,
                                 oldest=b, count=count, repeats=repeats, bypasses=bypasses,
                                 pct_repeats=round(100 * count / repeats, 6),
                                 pct_bypasses=round(100 * count / bypasses, 6)))
        if sum(previous.values()) != total:
            raise ValueError(f"Incorrect {workload} bypass total")
    write_csv(run / "gto_pairs.csv", rows)


if __name__ == "__main__":
    collect_pairs(Path(sys.argv[1]))
