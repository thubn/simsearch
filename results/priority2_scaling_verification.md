# Priority 2 Scaling Verification

Methods covered (all six, all three N values):
`float32_avx2`, `binary`, `two_step_RF10`, `two_step_mf_RF10`, `two_step_RF50`, `two_step_mf_RF50`.

The two RF=50 variants were added by `scripts/run_priority2_scaling_rf50.sh` after the original four-method run; rows for the other four methods are unchanged from that earlier run. Verification checks performed on the combined raw CSV: 18 (method, N) groups; 1000 rows per group; no NaN or negative timings; `num_survivors = k * RF = 5000` for every RF=50 row; mean NDCG@100 strictly larger for RF=50 than RF=10 at each N for both standard and MF variants; mean `T_rescore_ms` is ~4.5x-4.9x larger for RF=50 than RF=10 (expected ~5x = ratio of candidate set sizes); mean `T_binary_scan_ms` is RF-independent to within ~4% at the largest N (the relative spread inflates at N=60K because absolute values are sub-0.2 ms and dominated by measurement noise).

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

## two_step_RF50
- Total latency by N: N=60000: 1.89266 ms, N=300000: 3.09516 ms, N=1200000: 6.41932 ms
- Rescore latency by N: N=60000: 1.17153 ms, N=300000: 1.33968 ms, N=1200000: 1.36756 ms
- Accuracy by N: N=60000: ndcg=0.9957, jaccard=0.9939, N=300000: ndcg=0.9957, jaccard=0.9949, N=1200000: ndcg=0.9946, jaccard=0.9927

## two_step_mf_RF10
- Total latency by N: N=60000: 0.867103 ms, N=300000: 1.69479 ms, N=1200000: 4.95274 ms
- Rescore latency by N: N=60000: 0.621288 ms, N=300000: 0.672386 ms, N=1200000: 0.706132 ms
- Accuracy by N: N=60000: ndcg=0.9757, jaccard=0.9624, N=300000: ndcg=0.9706, jaccard=0.9525, N=1200000: ndcg=0.9688, jaccard=0.9487

## two_step_mf_RF50
- Total latency by N: N=60000: 3.48943 ms, N=300000: 4.8795 ms, N=1200000: 8.39096 ms
- Rescore latency by N: N=60000: 2.79166 ms, N=300000: 3.17458 ms, N=1200000: 3.22872 ms
- Accuracy by N: N=60000: ndcg=0.9808, jaccard=0.9745, N=300000: ndcg=0.9790, jaccard=0.9724, N=1200000: ndcg=0.9791, jaccard=0.9728

## Checklist
- Float32 and binary linearity should be assessed from the table above; no conclusion is forced by this script.
- Two-step Step 1 dominance should be assessed by comparing binary scan plus candidate selection against rescore plus final top-k.
- Unexpected trends should be reported in `docs/revision_experiment_results_summary.md` after the full run.
