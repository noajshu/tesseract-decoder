#!/usr/bin/env python3
import subprocess
import shutil
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_flamegraph_repo(path: Path):
    if path.exists():
        return
    repo_url = "https://github.com/brendangregg/FlameGraph.git"
    run(["git", "clone", repo_url, str(path)])


def main():
    if not shutil.which("perf"):
        sys.exit("perf not found. Please install linux-tools to use this script.")

    fg_path = Path("FlameGraph")
    ensure_flamegraph_repo(fg_path)

    run(["bazel", "build", "src:all"])

    perf_data = Path("perf.data")
    tesseract_cmd = [
        "./bazel-bin/src/tesseract",
        "--pqlimit", "200000",
        "--beam", "5",
        "--num-det-orders", "5",
        "--sample-num-shots", "100",
        "--det-order-seed", "13267562",
        "--circuit", "testdata/colorcodes/r=9,d=9,p=0.002,noise=si1000,c=superdense_color_code_X,q=121,gates=cz.stim",
        "--sample-seed", "717347",
        "--threads", "4",
        "--det-order-bfs",
    ]

    run(["perf", "record", "-F", "99", "-g", "-o", str(perf_data), "--"] + tesseract_cmd)

    with open("perf.unfold", "w") as out:
        run(["perf", "script", "-i", str(perf_data)], stdout=out)

    with open("perf.folded", "w") as out, open("perf.unfold") as inp:
        run([str(fg_path / "stackcollapse-perf.pl")], stdin=inp, stdout=out)

    with open("perf.svg", "w") as out, open("perf.folded") as inp:
        run([str(fg_path / "flamegraph.pl")], stdin=inp, stdout=out)

    print("Flamegraph written to perf.svg")


if __name__ == "__main__":
    main()
