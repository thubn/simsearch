#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/build_info

lscpu > results/build_info/lscpu.txt
free -h > results/build_info/memory.txt
uname -a > results/build_info/uname.txt
gcc --version > results/build_info/gcc_version.txt 2>&1 || true
clang --version > results/build_info/clang_version.txt 2>&1 || true
cmake --version > results/build_info/cmake_version.txt
git rev-parse HEAD > results/build_info/git_commit.txt
git status --short > results/build_info/git_status.txt
