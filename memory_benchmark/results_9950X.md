# 9950X memory bandwidth (DDR5, 64 GB)
Date: 2026-05-21
ARRAY_SIZE: 12 GB (12884901888 bytes, unchanged hard-coded default; 56 GB RAM available, no scaling needed)
Governor / turbo: performance governor (amd-pstate-epp); boost/turbo DISABLED via /sys/devices/system/cpu/cpufreq/boost = 0
Sequential read (AVX2):  min=49.22 median=49.81 max=50.08 GB/s (n=6)
Strided read (AVX2):     min=42.69 median=42.84 max=43.02 GB/s (n=6)

Sequential ceiling change vs 5950X (29.46 GB/s): 1.69x (+69.1%)
Strided ceiling change vs 5950X (36.71 GB/s):    1.17x (+16.7%)

## Notes
- NOTE ON MEMORY TYPE: the task brief described the testbed as "64 GB DDR4",
  but `dmidecode --type 17` reports 2x 32 GB DDR5-4800 (4800 MT/s, configured
  4800 MT/s, 1.1 V). The Ryzen 9 9950X (Zen 5) only supports DDR5, so the
  ceiling figures here are for DDR5-4800, not DDR4. §7.3 prose should reflect
  this.
- Host: meow1, Linux 6.17.0-19-generic, AMD Ryzen 9 9950X 16-Core (32 threads).
- Raw per-run results (GB/s):
    run  sequential  strided
      1     49.61     42.95
      2     49.75     42.88
      3     49.87     42.80
      4     49.22     43.02
      5     50.08     42.75
      6     49.92     42.69
- memory_benchmark.c unchanged; access pattern (NUM_ITERATIONS=16, NUM_STRIDES=8,
  STRIDE_LENGTH=1024 floats = 4 KB) identical to the 5950X measurement.
- Stability: median is within 1.2% of min and 0.6% of max (sequential), and
  within 0.4% of both min and max (strided) -- well inside the 10% bound.
- Host context captured in bandwidth_host.txt; DIMM details in bandwidth_dmi.txt.
