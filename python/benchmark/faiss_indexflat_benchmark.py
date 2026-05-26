"""faiss_indexflat_benchmark.py — one-off baseline for the TECS revision.

Measures FAISS IndexFlatIP (exact flat scan from a mature library) on the
same workload as the v3 9950X benchmark: 1.2M mxbai Wikipedia vectors,
313 queries, k=100, 10 inner repeats per query, one query timed at a time.

This is a sibling of benchmark_geomean.py, NOT a modification of it. It is not
intended to be reusable beyond this single experiment.

Outputs:
  python/out/benchmark_dim1024_k100_q_faiss_indexflat.json
  python/out/benchmark_dim1024_k100_q_faiss_indexflat.host.json
"""

import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from math import exp, log2

import numpy as np
import pyarrow.parquet as pq
import faiss

# --- paths -----------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARQUET = os.path.join(REPO, "python/out/wiki_mxbai_1024_N1200000_seed42.parquet")
QUERIES = os.path.join(REPO, "python/query_embeddings/combined.jsonl")
V3_JSON = os.path.join(REPO, "python/jupyter/results/benchmark_dim1024_k100_q.json")  # optional cross-check; skipped if absent
OUT_JSON = os.path.join(REPO, "python/out/benchmark_dim1024_k100_q_faiss_indexflat.json")
OUT_HOST = os.path.join(REPO, "python/out/benchmark_dim1024_k100_q_faiss_indexflat.host.json")

DIM = 1024
K = 100
INNER_REPEATS = 10


# --- helpers (NDCG / geomean copied from benchmark_geomean.py conventions) -------
def calculate_ndcg(ground_truth, prediction):
    """Same formula as benchmark_geomean.calculate_ndcg; entries are (score, idx, text)."""
    if not ground_truth or not prediction:
        return 0.0
    k = min(len(ground_truth), len(prediction))
    truth_positions = {idx: pos for pos, (_, idx, _) in enumerate(ground_truth[:k])}
    dcg = 0.0
    for i in range(k):
        pred_idx = prediction[i][1]
        if pred_idx in truth_positions:
            position_diff = abs(float(truth_positions[pred_idx] - i))
            relevance = exp(-position_diff / k)
            dcg += relevance / log2(i + 2)
    idcg = sum(1.0 / log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def _geomean(values):
    if len(values) == 0:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return float(np.exp(np.log(clipped).mean()))


def load_database():
    """Stack embedding_0..embedding_1023 into a (N, 1024) C-contiguous float32 array."""
    emb_cols = [f"embedding_{i}" for i in range(DIM)]
    print(f"Reading parquet {os.path.basename(PARQUET)} ...", flush=True)
    table = pq.read_table(PARQUET, columns=emb_cols).combine_chunks()
    n = table.num_rows
    db = np.empty((n, DIM), dtype=np.float32)
    for j, col in enumerate(emb_cols):
        db[:, j] = table.column(col).to_numpy(zero_copy_only=False)
    del table
    db = np.ascontiguousarray(db, dtype=np.float32)
    print(f"Database loaded: shape={db.shape}, dtype={db.dtype}, "
          f"{db.nbytes / 2**30:.2f} GiB", flush=True)
    return db


def load_queries():
    rows = []
    with open(QUERIES) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    q = np.array([r["embedding"] for r in rows], dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    print(f"Queries loaded: shape={q.shape}, dtype={q.dtype}", flush=True)
    return q, rows


def time_search(index, queries):
    """Run the v3-style timing loop: per query, INNER_REPEATS timed searches.

    Returns (times_ms list of len num_queries*INNER_REPEATS, top100_ids array).
    One query timed at a time as a (1, DIM) contiguous float32 buffer.
    """
    num_q = queries.shape[0]
    times_ms = []
    top100 = np.empty((num_q, K), dtype=np.int64)
    # warm-up (not recorded)
    for w in range(3):
        index.search(np.ascontiguousarray(queries[0:1]), K)
    for qi in range(num_q):
        q = np.ascontiguousarray(queries[qi:qi + 1], dtype=np.float32)
        last_ids = None
        for _ in range(INNER_REPEATS):
            t0 = time.perf_counter()
            _, ids = index.search(q, K)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            last_ids = ids
        top100[qi] = last_ids[0]
    return times_ms, top100


def exact_ground_truth(db, queries):
    """Independent exact top-K via numpy inner-product brute force."""
    num_q = queries.shape[0]
    gt = np.empty((num_q, K), dtype=np.int64)
    for qi in range(num_q):
        scores = db @ queries[qi]
        part = np.argpartition(scores, -K)[-K:]
        gt[qi] = part[np.argsort(scores[part])[::-1]]
    return gt


def summarize(times_ms):
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "geomean_ms": _geomean(arr),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "num_samples": int(arr.size),
    }


def main():
    print(f"faiss version: {faiss.__version__}", flush=True)
    logical = os.cpu_count() or 1
    try:
        smt_on = open("/sys/devices/system/cpu/smt/active").read().strip() == "1"
    except OSError:
        smt_on = False
    physical = logical // 2 if smt_on else logical
    print(f"logical cores={logical}, smt={'on' if smt_on else 'off'}, "
          f"physical cores used for multi-thread={physical}", flush=True)

    db = load_database()
    queries, _ = load_queries()
    assert db.shape[1] == DIM and queries.shape[1] == DIM

    print("Building IndexFlatIP and adding database vectors ...", flush=True)
    index = faiss.IndexFlatIP(DIM)
    index.add(db)
    assert index.ntotal == db.shape[0]

    # --- single-thread timing (headline) ---
    faiss.omp_set_num_threads(1)
    print("Timing single-thread (omp_num_threads=1) ...", flush=True)
    t_start = time.perf_counter()
    st_times, st_top100 = time_search(index, queries)
    print(f"  single-thread loop done in {time.perf_counter() - t_start:.1f}s", flush=True)

    # --- multi-thread timing (secondary, prose caveat only) ---
    faiss.omp_set_num_threads(physical)
    print(f"Timing multi-thread (omp_num_threads={physical}) ...", flush=True)
    t_start = time.perf_counter()
    mt_times, mt_top100 = time_search(index, queries)
    print(f"  multi-thread loop done in {time.perf_counter() - t_start:.1f}s", flush=True)
    faiss.omp_set_num_threads(1)

    # --- exact-ness verification (Step 3) ---
    print("Computing independent numpy exact ground truth ...", flush=True)
    gt = exact_ground_truth(db, queries)
    num_q = queries.shape[0]

    set_equal = 0
    jaccards = []
    ndcgs = []
    for qi in range(num_q):
        faiss_ids = st_top100[qi]
        gt_ids = gt[qi]
        sf, sg = set(faiss_ids.tolist()), set(gt_ids.tolist())
        if sf == sg:
            set_equal += 1
        inter = len(sf & sg)
        union = len(sf | sg)
        jaccards.append(inter / union if union else 1.0)
        # NDCG: entries (score, idx, text); score unused by the formula
        gt_entries = [(0.0, int(i), "") for i in gt_ids]
        pred_entries = [(0.0, int(i), "") for i in faiss_ids]
        ndcgs.append(calculate_ndcg(gt_entries, pred_entries))
    mean_ndcg = float(np.mean(ndcgs))
    mean_jaccard = float(np.mean(jaccards))
    print(f"  queries with FAISS top-100 == numpy exact top-100: {set_equal}/{num_q}", flush=True)
    print(f"  mean NDCG@100={mean_ndcg:.6f}  mean Jaccard@100={mean_jaccard:.6f}", flush=True)

    # cross-check against the repo's v3 float baseline (only top-5 stored there)
    top5_equal = None
    if os.path.exists(V3_JSON):
        v3 = json.load(open(V3_JSON))
        float_top5 = []
        for run in v3["runs"]:
            for s in run["searches"]:
                if s["method"] == "float":
                    float_top5.append([int(e[1]) for e in s["metrics"]["results"]])
                    break
        if len(float_top5) == num_q:
            top5_equal = sum(
                1 for qi in range(num_q)
                if set(st_top100[qi][:5].tolist()) == set(float_top5[qi])
            )
            print(f"  queries with FAISS top-5 == v3 float baseline top-5: "
                  f"{top5_equal}/{num_q} (v3 JSON only stores top-5)", flush=True)

    # --- assemble JSON ---
    mt_summary = summarize(mt_times)
    st_summary = summarize(st_times)
    # the verification metrics belong to the exact method regardless of threading
    for s in (st_summary, mt_summary):
        s["ndcg_at_100"] = round(mean_ndcg, 6)
        s["jaccard"] = round(mean_jaccard, 6)

    out = {
        "metadata": {
            "library": "faiss-cpu",
            "library_version": faiss.__version__,
            "method": "IndexFlatIP",
            "num_vectors": int(index.ntotal),
            "vector_dim": DIM,
            "k": K,
            "num_queries": num_q,
            "inner_repeats": INNER_REPEATS,
            "single_thread": {"omp_num_threads": 1},
            "multi_thread": {
                "omp_num_threads": physical,
                "median_ms": mt_summary["median_ms"],
            },
            "host": socket.gethostname(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "embedding_file": os.path.relpath(PARQUET, REPO),
            "query_file": os.path.relpath(QUERIES, REPO),
            "verification": {
                "queries_top100_equal_numpy_exact": set_equal,
                "queries_top5_equal_v3_float_baseline": top5_equal,
                "v3_json_note": "benchmark_dim1024_k100_q.json stores only top-5 "
                                "per query; top-100 set check is against an "
                                "independent numpy brute-force inner-product "
                                "ground truth (equivalent to the exact float scan).",
            },
        },
        "summary": {
            "single_thread": st_summary,
            "multi_thread": mt_summary,
        },
        "method_stats": {
            "single_thread": {"times_ms": st_times},
            "multi_thread": {"times_ms": mt_times},
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}", flush=True)

    # --- host sidecar ---
    def _run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            return f"<{e}>"

    host = {
        "hostname": _run("hostname").strip(),
        "uname_a": _run("uname -a").strip(),
        "lscpu_head20": _run("lscpu | head -20").strip(),
        "python": platform.python_version(),
        "faiss_version": faiss.__version__,
    }
    with open(OUT_HOST, "w") as f:
        json.dump(host, f, indent=2)
    print(f"Wrote {OUT_HOST}", flush=True)

    # --- four-line report ---
    scalar_float_ref = 908.30
    speedup = scalar_float_ref / st_summary["geomean_ms"]
    print()
    print(f"FAISS IndexFlatIP, N={index.ntotal}, k={K}, "
          f"{num_q} queries x {INNER_REPEATS} inner repeats:")
    print(f"  single-thread:  geomean = {st_summary['geomean_ms']:.2f} ms, "
          f"median = {st_summary['median_ms']:.2f} ms, "
          f"p95 = {st_summary['p95_ms']:.2f} ms")
    print(f"  multi-thread ({physical} threads):  median = {mt_summary['median_ms']:.2f} ms")
    print(f"  NDCG@100 = {mean_ndcg:.4f}, Jaccard = {mean_jaccard:.4f}")
    print(f"  Speedup vs scalar float ({scalar_float_ref} ms reference from v3): "
          f"{speedup:.1f} x")


if __name__ == "__main__":
    main()
