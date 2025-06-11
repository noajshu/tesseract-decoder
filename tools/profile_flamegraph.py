#!/usr/bin/env python3
import os
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
    # Allow custom perf binary via PERF_BINARY environment variable (e.g. perf5)
    perf_bin = os.environ.get("PERF_BINARY", "perf")
    if not shutil.which(perf_bin):
        sys.exit(
            f"{perf_bin} not found. Please install linux-tools or set PERF_BINARY"
        )

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

    run([perf_bin, "record", "-F", "99", "-g", "-o", str(perf_data), "--"] + tesseract_cmd)

    with open("perf.unfold", "w") as out:
        run([perf_bin, "script", "-i", str(perf_data)], stdout=out)

    with open("perf.folded", "w") as out, open("perf.unfold") as inp:
        run([str(fg_path / "stackcollapse-perf.pl")], stdin=inp, stdout=out)

    with open("perf.svg", "w") as out, open("perf.folded") as inp:
        run([str(fg_path / "flamegraph.pl")], stdin=inp, stdout=out)

    with open("decode_to_errors.annotate", "w") as out:
        run(
            [
                perf_bin,
                "annotate",
                "-i",
                str(perf_data),
                "-l",
                "--stdio",
                "decode_to_errors",
            ],
            stdout=out,
        )

    print("Flamegraph written to perf.svg")
    print("Line-level profile written to decode_to_errors.annotate")


if __name__ == "__main__":
    main()
