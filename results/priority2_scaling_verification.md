# Priority 2 Scaling Verification

## binary
- Total latency by N: N=60000: 0.194874 ms, N=300000: 0.94471 ms, N=1200000: 4.28602 ms
- Rescore latency by N: N=60000: 1e-12 ms, N=300000: 1e-12 ms, N=1200000: 1e-12 ms
- Accuracy by N: N=60000: ndcg=0.5949, jaccard=0.4602, N=300000: ndcg=0.5799, jaccard=0.4413, N=1200000: ndcg=0.5816, jaccard=0.4415

## float32_avx2
- Total latency by N: N=60000: 5.62863 ms, N=300000: 27.7707 ms, N=1200000: 107.864 ms
- Rescore latency by N: N=60000: 1e-12 ms, N=300000: 1e-12 ms, N=1200000: 1e-12 ms
- Accuracy by N: N=60000: ndcg=0.9959, jaccard=0.9943, N=300000: ndcg=0.9961, jaccard=0.9956, N=1200000: ndcg=0.9949, jaccard=0.9933

## two_step_RF10
- Total latency by N: N=60000: 0.495354 ms, N=300000: 1.28897 ms, N=1200000: 4.52461 ms
- Rescore latency by N: N=60000: 0.241978 ms, N=300000: 0.273308 ms, N=1200000: 0.293386 ms
- Accuracy by N: N=60000: ndcg=0.9886, jaccard=0.9781, N=300000: ndcg=0.9844, jaccard=0.9703, N=1200000: ndcg=0.9813, jaccard=0.9635

## two_step_mf_RF10
- Total latency by N: N=60000: 0.867103 ms, N=300000: 1.69479 ms, N=1200000: 4.95274 ms
- Rescore latency by N: N=60000: 0.621288 ms, N=300000: 0.672386 ms, N=1200000: 0.706132 ms
- Accuracy by N: N=60000: ndcg=0.9757, jaccard=0.9624, N=300000: ndcg=0.9706, jaccard=0.9525, N=1200000: ndcg=0.9688, jaccard=0.9487

## Checklist
- Float32 and binary linearity should be assessed from the table above; no conclusion is forced by this script.
- Two-step Step 1 dominance should be assessed by comparing binary scan plus candidate selection against rescore plus final top-k.
- Unexpected trends should be reported in `docs/revision_experiment_results_summary.md` after the full run.
